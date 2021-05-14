from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import tensorflow as tf


class E2EFS_SGD(optimizers.SGD):

    def __init__(self, e2efs_layer, th=.1, adam_lr=0.01, beta_1=0.5, beta_2=0.999,
                 amsgrad=False, **kwargs):
        super(E2EFS_SGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.adam_lr = K.variable(adam_lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
        self.amsgrad = amsgrad
        self.e2efs_layer = e2efs_layer
        self.th = th

    def get_gradients(self, loss, params):
        grads = super(E2EFS_SGD, self).get_gradients(loss, params)
        if not(hasattr(self.e2efs_layer, 'regularization_loss')):
            return grads
        e2efs_grad = grads[0]
        e2efs_regularizer_grad = K.gradients(self.e2efs_layer.regularization_loss, [self.e2efs_layer.kernel])[0]
        # norm_e2efs_grad_clipped = K.maximum(0.1, tf.norm(e2efs_grad) + K.epsilon())
        # e2efs_regularizer_grad_corrected = e2efs_regularizer_grad / K.max(K.abs(e2efs_regularizer_grad) + K.epsilon())
        # e2efs_grad_corrected = e2efs_grad / K.max(K.abs(e2efs_grad) + K.epsilon())
        e2efs_regularizer_grad_corrected = e2efs_regularizer_grad / (K.tf.norm(e2efs_regularizer_grad) + K.epsilon())
        e2efs_grad_corrected = e2efs_grad / (K.tf.norm(e2efs_grad) + K.epsilon())
        # e2efs_regularizer_grad_corrected = norm_e2efs_grad_clipped * e2efs_regularizer_grad / (tf.norm(e2efs_regularizer_grad) + K.epsilon())
        combined_e2efs_grad = (1. - self.e2efs_layer.moving_factor) * e2efs_grad_corrected + \
                              self.e2efs_layer.moving_factor * e2efs_regularizer_grad_corrected
        # combined_e2efs_grad_norm = tf.norm(combined_e2efs_grad) + K.epsilon()
        # combined_e2efs_grad = optimizers.clip_norm(combined_e2efs_grad, self.e2efs_norm_max, combined_e2efs_grad_norm)
        # combined_e2efs_grad = K.maximum(combined_e2efs_grad_norm, self.e2efs_norm_min) / combined_e2efs_grad_norm * combined_e2efs_grad
        combined_e2efs_grad = K.sign(
            self.e2efs_layer.moving_factor) * K.minimum(self.th, K.max(
            K.abs(combined_e2efs_grad))) * combined_e2efs_grad / K.max(
            K.abs(combined_e2efs_grad) + K.epsilon())
        # combined_e2efs_grad = K.tf.Print(combined_e2efs_grad, [K.max(combined_e2efs_grad), K.min(combined_e2efs_grad)])
        grads[0] = combined_e2efs_grad
        return grads
        # e2efs_grad = grads[0]
        # e2efs_regularizer_grad = K.gradients(self.e2efs_layer.regularization_loss, [self.e2efs_layer.kernel])[0]
        # # norm_e2efs_grad_clipped = K.maximum(0.1, tf.norm(e2efs_grad) + K.epsilon())
        # # e2efs_regularizer_grad_corrected = e2efs_regularizer_grad / K.max(K.abs(e2efs_regularizer_grad) + K.epsilon())
        # # e2efs_grad_corrected = e2efs_grad / K.max(K.abs(e2efs_grad) + K.epsilon())
        # e2efs_regularizer_grad_corrected = e2efs_regularizer_grad / (K.tf.norm(e2efs_regularizer_grad) + K.epsilon())
        # e2efs_grad_corrected = e2efs_grad / (K.tf.norm(e2efs_grad) + K.epsilon())
        # # e2efs_regularizer_grad_corrected = norm_e2efs_grad_clipped * e2efs_regularizer_grad / (tf.norm(e2efs_regularizer_grad) + K.epsilon())
        # combined_e2efs_grad = (1. - self.e2efs_layer.moving_factor) * e2efs_grad_corrected + \
        #                       self.e2efs_layer.moving_factor * e2efs_regularizer_grad_corrected
        # # combined_e2efs_grad_norm = tf.norm(combined_e2efs_grad) + K.epsilon()
        # # combined_e2efs_grad = optimizers.clip_norm(combined_e2efs_grad, self.e2efs_norm_max, combined_e2efs_grad_norm)
        # # combined_e2efs_grad = K.maximum(combined_e2efs_grad_norm, self.e2efs_norm_min) / combined_e2efs_grad_norm * combined_e2efs_grad
        # combined_e2efs_grad = K.sign(self.e2efs_layer.moving_factor) * self.e2efs_norm_min * combined_e2efs_grad / K.max(K.abs(combined_e2efs_grad) + K.epsilon())
        # grads[0] = combined_e2efs_grad
        # return grads
        # grads = super(E2EFS_SGD, self).get_gradients(loss, params)
        # if not (hasattr(self.e2efs_layer, 'regularization_loss')):
        #     return grads
        # e2efs_grad = grads[0]
        # e2efs_regularizer_grad = K.gradients(self.e2efs_layer.regularization_loss, [self.e2efs_layer.kernel])[0]
        # norm_e2efs_grad_clipped = K.maximum(.1, tf.norm(e2efs_grad) + K.epsilon())
        # e2efs_regularizer_grad_corrected = norm_e2efs_grad_clipped * e2efs_regularizer_grad / (
        #         tf.norm(e2efs_regularizer_grad) + K.epsilon())
        # combined_e2efs_grad = (1. - self.e2efs_layer.moving_factor) * e2efs_grad + \
        #                       self.e2efs_layer.moving_factor * e2efs_regularizer_grad_corrected
        # combined_e2efs_grad = optimizers.clip_norm(combined_e2efs_grad, self.e2efs_norm_max,
        #                                            tf.norm(combined_e2efs_grad) + K.epsilon())
        # # combined_e2efs_grad = K.tf.Print(combined_e2efs_grad, [K.min(combined_e2efs_grad), K.max(combined_e2efs_grad), self.e2efs_layer.moving_factor])
        # combined_e2efs_grad = K.sign(self.e2efs_layer.moving_factor) * combined_e2efs_grad
        #
        # grads[0] = combined_e2efs_grad
        # return grads

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        adam_lr = self.adam_lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
            adam_lr = adam_lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        adam_lr_t = adam_lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.ms = K.zeros(K.int_shape(params[0]), dtype=K.dtype(params[0]))
        self.vs = K.zeros(K.int_shape(params[0]), dtype=K.dtype(params[0]))
        self.weights = [self.iterations] + moments + vhats + [self.ms] + [self.vs]
        for i, (p, g, m, vhat) in enumerate(zip(params, grads, moments, vhats)):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            if i == 0 and self.e2efs_layer is not None:
                nnz = K.sum(K.cast(K.greater(p, 0.), K.floatx()))
                m_t = (self.beta_1 * self.ms) + (1. - self.beta_1) * g
                v_t = (self.beta_2 * self.vs) + (1. - self.beta_2) * K.square(g)
                if self.amsgrad:
                    vhat_t = K.maximum(vhat, v_t)
                    p_t = p - adam_lr_t * m_t / (K.sqrt(vhat_t) + K.epsilon())
                    self.updates.append(K.update(vhat, vhat_t))
                else:
                    p_t = p - adam_lr_t * m_t / (K.sqrt(v_t) + K.epsilon())

                self.updates.append(K.update(self.ms, m_t))
                self.updates.append(K.update(self.vs, v_t))
                new_p = K.switch(K.less_equal(nnz, self.e2efs_layer.units), new_p, p_t)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates


class E2EFS_Adam(optimizers.Adam):

    def __init__(self, e2efs_layer, th=.1, lr_momentum=False, **kwargs):
        super(E2EFS_Adam, self).__init__(**kwargs)
        self.e2efs_layer = e2efs_layer
        self.th = th
        self.lr_momentum = lr_momentum

    def get_gradients(self, loss, params):
        grads = super(E2EFS_Adam, self).get_gradients(loss, params)
        if not (hasattr(self.e2efs_layer, 'regularization_loss')):
            return grads
        e2efs_grad = grads[0]
        e2efs_regularizer_grad = K.gradients(self.e2efs_layer.regularization_loss, [self.e2efs_layer.kernel])[0]
        # norm_e2efs_grad_clipped = K.maximum(0.1, tf.norm(e2efs_grad) + K.epsilon())
        # e2efs_regularizer_grad_corrected = e2efs_regularizer_grad / K.max(K.abs(e2efs_regularizer_grad) + K.epsilon())
        # e2efs_grad_corrected = e2efs_grad / K.max(K.abs(e2efs_grad) + K.epsilon())
        e2efs_regularizer_grad_corrected = e2efs_regularizer_grad / (K.tf.norm(e2efs_regularizer_grad) + K.epsilon())
        e2efs_grad_corrected = e2efs_grad / (K.tf.norm(e2efs_grad) + K.epsilon())
        # e2efs_regularizer_grad_corrected = norm_e2efs_grad_clipped * e2efs_regularizer_grad / (tf.norm(e2efs_regularizer_grad) + K.epsilon())
        combined_e2efs_grad = (1. - self.e2efs_layer.moving_factor) * e2efs_grad_corrected + \
                              self.e2efs_layer.moving_factor * e2efs_regularizer_grad_corrected
        # combined_e2efs_grad_norm = tf.norm(combined_e2efs_grad) + K.epsilon()
        # combined_e2efs_grad = optimizers.clip_norm(combined_e2efs_grad, self.e2efs_norm_max, combined_e2efs_grad_norm)
        # combined_e2efs_grad = K.maximum(combined_e2efs_grad_norm, self.e2efs_norm_min) / combined_e2efs_grad_norm * combined_e2efs_grad
        combined_e2efs_grad = K.sign(
            self.e2efs_layer.moving_factor) * K.minimum(self.th, K.max(K.abs(combined_e2efs_grad))) * combined_e2efs_grad / K.max(
            K.abs(combined_e2efs_grad) + K.epsilon())
        # combined_e2efs_grad = K.tf.Print(combined_e2efs_grad, [K.max(combined_e2efs_grad), K.min(combined_e2efs_grad)])
        grads[0] = combined_e2efs_grad
        return grads

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for i, (p, g, m, v, vhat) in enumerate(zip(params, grads, ms, vs, vhats)):
            beta_1 = 0.5 if i == 0 and self.e2efs_layer is not None and not self.lr_momentum else self.beta_1
            beta_2 = 0. if i == -1 and self.e2efs_layer is not None and not self.lr_momentum else self.beta_2
            m_t = (beta_1 * m) + (1. - beta_1) * g
            v_t = (beta_2 * v) + (1. - beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates
