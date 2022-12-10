from keras import optimizers
from keras import backend as K
import tensorflow as tf
from tensorflow.python.training import gen_training_ops
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.framework import ops


def get_gradients(self, tape, loss, var_list, grad_loss=None):
        """Called in `minimize` to compute gradients from loss."""
        grads = tape.gradient(loss, var_list, grad_loss)
        if not (hasattr(self.e2efs_layer, 'regularization_loss')):
            return list(zip(grads, var_list))
        with tf.GradientTape() as e2efs_tape:
            e2efs_loss = self.e2efs_layer.regularization_func(self.e2efs_layer.kernel)
        e2efs_grad = grads[0]
        e2efs_regularizer_grad = e2efs_tape.gradient(e2efs_loss, [self.e2efs_layer.kernel])[0]
        # tf.print(e2efs_regularizer_grad)
        e2efs_regularizer_grad_corrected = e2efs_regularizer_grad / (tf.norm(e2efs_regularizer_grad) + K.epsilon())
        e2efs_grad_corrected = e2efs_grad / (tf.norm(e2efs_grad) + K.epsilon())
        combined_e2efs_grad = (1. - self.e2efs_layer.moving_factor) * e2efs_grad_corrected + \
                              self.e2efs_layer.moving_factor * e2efs_regularizer_grad_corrected
        combined_e2efs_grad = K.sign(
            self.e2efs_layer.moving_factor) * K.minimum(self.th, K.max(
            K.abs(combined_e2efs_grad))) * combined_e2efs_grad / K.max(
            K.abs(combined_e2efs_grad) + K.epsilon())
        grads[0] = combined_e2efs_grad
        return list(zip(grads, var_list))


def resource_apply_dense_e2efs(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'e2efs_m')
        v = self.get_slot(var, 'e2efs_v')

        if not self.amsgrad:
            return gen_training_ops.ResourceApplyAdam(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                beta1_power=coefficients['e2efs_beta_1_power'],
                beta2_power=coefficients['e2efs_beta_2_power'],
                lr=coefficients['e2efs_lr_t'],
                beta1=coefficients['e2efs_beta_1_t'],
                beta2=coefficients['e2efs_beta_2_t'],
                epsilon=coefficients['epsilon'],
                grad=grad,
                use_locking=self._use_locking)
        else:
            vhat = self.get_slot(var, 'e2efs_vhat')
            return gen_training_ops.ResourceApplyAdamWithAmsgrad(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                vhat=vhat.handle,
                beta1_power=coefficients['e2efs_beta_1_power'],
                beta2_power=coefficients['e2efs_beta_2_power'],
                lr=coefficients['e2efs_lr_t'],
                beta1=coefficients['e2efs_beta_1_t'],
                beta2=coefficients['e2efs_beta_2_t'],
                epsilon=coefficients['epsilon'],
                grad=grad,
                use_locking=self._use_locking)



class E2EFS_SGD(optimizers.SGD):

    def __init__(self, e2efs_layer, th=.1, e2efs_lr=0.01, beta_1=0.5, beta_2=0.999,
                 amsgrad=False, **kwargs):
        super(E2EFS_SGD, self).__init__(**kwargs)
        self._set_hyper('e2efs_lr', e2efs_lr)
        self._set_hyper('e2efs_beta_1', beta_1)
        self._set_hyper('e2efs_beta_2', beta_2)
        self.epsilon = K.epsilon()
        self.amsgrad = amsgrad
        self.e2efs_layer = e2efs_layer
        self.th = th

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        super(E2EFS_SGD, self)._create_slots(var_list)
        for var in var_list:
            if 'e2efs' in var.name:
                self.add_slot(var, 'e2efs_m')
        for var in var_list:
            if 'e2efs' in var.name:
                self.add_slot(var, 'e2efs_v')
        if self.amsgrad:
            for var in var_list:
                if 'e2efs' in var.name:
                    self.add_slot(var, 'e2efs_vhat')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(E2EFS_SGD, self)._prepare_local(var_device, var_dtype, apply_state)
        e2efs_lr_t = array_ops.identity(self._get_hyper('e2efs_lr', var_dtype))
        apply_state[(var_device, var_dtype)]["e2efs_lr_t"] = e2efs_lr_t

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('e2efs_beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('e2efs_beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        e2efs_lr = (apply_state[(var_device, var_dtype)]['e2efs_lr_t'] *
              (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                e2efs_lr=e2efs_lr,
                epsilon=ops.convert_to_tensor_v2_with_dispatch(
                    self.epsilon, var_dtype),
                e2efs_beta_1_t=beta_1_t,
                e2efs_beta_1_power=beta_1_power,
                e2efs_one_minus_beta_1_t=1 - beta_1_t,
                e2efs_beta_2_t=beta_2_t,
                e2efs_beta_2_power=beta_2_power,
                e2efs_one_minus_beta_2_t=1 - beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(E2EFS_SGD, self).set_weights(weights)

    def _get_gradients(self, tape, loss, var_list, grad_loss=None):
        return get_gradients(self, tape, loss, var_list, grad_loss)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if 'e2efs' in var.name:
            return resource_apply_dense_e2efs(self, grad, var, apply_state)
        else:
            return super(E2EFS_SGD, self)._resource_apply_dense(grad, var, apply_state)


class E2EFS_Adam(optimizers.Adam):

    def __init__(self, e2efs_layer, th=.1, e2efs_lr=0.01, beta_1=0.5, beta_2=0.999,
                 amsgrad=False, **kwargs):
        super(E2EFS_Adam, self).__init__(**kwargs)
        self._set_hyper('e2efs_lr', e2efs_lr)
        self._set_hyper('e2efs_beta_1', beta_1)
        self._set_hyper('e2efs_beta_2', beta_2)
        self.epsilon = K.epsilon()
        self.amsgrad = amsgrad
        self.e2efs_layer = e2efs_layer
        self.th = th

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        super(E2EFS_Adam, self)._create_slots(var_list)
        for var in var_list:
            if 'e2efs' in var.name:
                self.add_slot(var, 'e2efs_m')
        for var in var_list:
            if 'e2efs' in var.name:
                self.add_slot(var, 'e2efs_v')
        if self.amsgrad:
            for var in var_list:
                if 'e2efs' in var.name:
                    self.add_slot(var, 'e2efs_vhat')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(E2EFS_Adam, self)._prepare_local(var_device, var_dtype, apply_state)
        e2efs_lr_t = array_ops.identity(self._get_hyper('e2efs_lr', var_dtype))
        apply_state[(var_device, var_dtype)]["e2efs_lr_t"] = e2efs_lr_t

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('e2efs_beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('e2efs_beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        e2efs_lr = (apply_state[(var_device, var_dtype)]['e2efs_lr_t'] *
              (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                e2efs_lr=e2efs_lr,
                epsilon=ops.convert_to_tensor_v2_with_dispatch(
                    self.epsilon, var_dtype),
                e2efs_beta_1_t=beta_1_t,
                e2efs_beta_1_power=beta_1_power,
                e2efs_one_minus_beta_1_t=1 - beta_1_t,
                e2efs_beta_2_t=beta_2_t,
                e2efs_beta_2_power=beta_2_power,
                e2efs_one_minus_beta_2_t=1 - beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(E2EFS_Adam, self).set_weights(weights)

    def _get_gradients(self, tape, loss, var_list, grad_loss=None):
        return get_gradients(self, tape, loss, var_list, grad_loss)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if 'e2efs' in var.name:
            return resource_apply_dense_e2efs(self, grad, var, apply_state)
        else:
            return super(E2EFS_Adam, self)._resource_apply_dense(grad, var, apply_state)


class E2EFS_RMSprop(optimizers.RMSprop):

    def __init__(self, e2efs_layer, th=.1, e2efs_lr=0.01, beta_1=0.5, beta_2=0.999,
                 amsgrad=False, **kwargs):
        super(E2EFS_RMSprop, self).__init__(**kwargs)
        self._set_hyper('e2efs_lr', e2efs_lr)
        self._set_hyper('e2efs_beta_1', beta_1)
        self._set_hyper('e2efs_beta_2', beta_2)
        self.epsilon = K.epsilon()
        self.amsgrad = amsgrad
        self.e2efs_layer = e2efs_layer
        self.th = th

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        super(E2EFS_RMSprop, self)._create_slots(var_list)
        for var in var_list:
            if 'e2efs' in var.name:
                self.add_slot(var, 'e2efs_m')
        for var in var_list:
            if 'e2efs' in var.name:
                self.add_slot(var, 'e2efs_v')
        if self.amsgrad:
            for var in var_list:
                if 'e2efs' in var.name:
                    self.add_slot(var, 'e2efs_vhat')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(E2EFS_RMSprop, self)._prepare_local(var_device, var_dtype, apply_state)
        e2efs_lr_t = array_ops.identity(self._get_hyper('e2efs_lr', var_dtype))
        apply_state[(var_device, var_dtype)]["e2efs_lr_t"] = e2efs_lr_t

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('e2efs_beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('e2efs_beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        e2efs_lr = (apply_state[(var_device, var_dtype)]['e2efs_lr_t'] *
              (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                e2efs_lr=e2efs_lr,
                epsilon=ops.convert_to_tensor_v2_with_dispatch(
                    self.epsilon, var_dtype),
                e2efs_beta_1_t=beta_1_t,
                e2efs_beta_1_power=beta_1_power,
                e2efs_one_minus_beta_1_t=1 - beta_1_t,
                e2efs_beta_2_t=beta_2_t,
                e2efs_beta_2_power=beta_2_power,
                e2efs_one_minus_beta_2_t=1 - beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(E2EFS_RMSprop, self).set_weights(weights)

    def _get_gradients(self, tape, loss, var_list, grad_loss=None):
        return get_gradients(self, tape, loss, var_list, grad_loss)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if 'e2efs' in var.name:
            return resource_apply_dense_e2efs(self, grad, var, apply_state)
        else:
            return super(E2EFS_RMSprop, self)._resource_apply_dense(grad, var, apply_state)
