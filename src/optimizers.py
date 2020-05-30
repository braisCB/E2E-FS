from keras import optimizers
from keras import backend as K
import tensorflow as tf


class E2EFS_SGD(optimizers.SGD):

    def __init__(self, e2efs_layer, e2efs_clipnorm=1., **kwargs):
        super(E2EFS_SGD, self).__init__(**kwargs)
        self.e2efs_layer = e2efs_layer
        self.e2efs_clipnorm = e2efs_clipnorm

    def get_gradients(self, loss, params):
        grads = super(E2EFS_SGD, self).get_gradients(loss, params)
        if not(hasattr(self.e2efs_layer, 'regularization_loss')):
            return grads
        e2efs_grad = grads[0]
        e2efs_regularizer_grad = K.gradients(self.e2efs_layer.regularization_loss[0], [self.e2efs_layer.kernel])[0]
        norm_e2efs_grad_clipped = K.maximum(0.1, tf.norm(e2efs_grad) + K.epsilon())
        e2efs_regularizer_grad_corrected = norm_e2efs_grad_clipped * e2efs_regularizer_grad / (tf.norm(e2efs_regularizer_grad) + K.epsilon())
        combined_e2efs_grad = (1. - self.e2efs_layer.moving_factor[1]) * e2efs_grad + \
                              self.e2efs_layer.moving_factor[1] * e2efs_regularizer_grad_corrected
        combined_e2efs_grad = optimizers.clip_norm(combined_e2efs_grad, self.e2efs_clipnorm, tf.norm(combined_e2efs_grad) + K.epsilon())
        combined_e2efs_grad = self.e2efs_layer.moving_factor[0] * combined_e2efs_grad
        grads[0] = combined_e2efs_grad
        return grads


class E2EFS_Adam(optimizers.Adam):

    def __init__(self, e2efs_layer, e2efs_clipnorm=1., **kwargs):
        super(E2EFS_Adam, self).__init__(**kwargs)
        self.e2efs_layer = e2efs_layer
        self.e2efs_clipnorm = e2efs_clipnorm

    def get_gradients(self, loss, params):
        grads = super(E2EFS_Adam, self).get_gradients(loss, params)
        if not (hasattr(self.e2efs_layer, 'regularization_loss')):
            return grads
        e2efs_grad = grads[0]
        e2efs_regularizer_grad = K.gradients(self.e2efs_layer.regularization_loss[0], [self.e2efs_layer.kernel])[0]
        norm_e2efs_grad_clipped = K.maximum(0.1, tf.norm(e2efs_grad) + K.epsilon())
        e2efs_regularizer_grad_corrected = norm_e2efs_grad_clipped * e2efs_regularizer_grad / (
                    tf.norm(e2efs_regularizer_grad) + K.epsilon())
        combined_e2efs_grad = (1. - self.e2efs_layer.moving_factor[1]) * e2efs_grad + \
                              self.e2efs_layer.moving_factor[1] * e2efs_regularizer_grad_corrected
        combined_e2efs_grad = optimizers.clip_norm(combined_e2efs_grad, self.e2efs_clipnorm,
                                                   tf.norm(combined_e2efs_grad) + K.epsilon())
        combined_e2efs_grad = self.e2efs_layer.moving_factor[0] * combined_e2efs_grad
        grads[0] = combined_e2efs_grad
        return grads
