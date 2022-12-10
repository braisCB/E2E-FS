from keras import backend as K
from keras.layers import Input
import tensorflow as tf


def clip(min_value, max_value):
    @tf.custom_gradient
    def clip_by_value(x):
        # x_clip = K.clip(x - bias, -2., 2.)
        s = K.clip(x, min_value, max_value)

        def grad(dy):
            return dy

        return s, grad
    return clip_by_value


def get_saliency(loss_func, model, beta=1.0, normalize=False, reduce_func='sum', use_abs=True):
    y_test = Input(shape=model.output_shape[1:])
    y_pred = model.output
    gain_function = get_gain_function(loss_func, beta, y_test, y_pred)
    gradient = K.gradients(gain_function, model.input)[0]
    # factor = K.sign(gradient) * K.sign(model.input)
    if use_abs:
        gradient = K.abs(gradient) # * factor
    # gradient = gradient / K.maximum(1e-6, K.abs(model.input))
    if normalize:
        axis = tuple(range(1, len(model.input_shape)))
        l1_norm = K.maximum(1e-6, K.sum(gradient, axis=axis, keepdims=True))
        factor = K.reshape(get_factor(loss_func, y_test, y_pred), K.shape(l1_norm))
        gradient *= factor / l1_norm
    saliency = gradient
    if reduce_func is not None:
        saliency = getattr(K, reduce_func)(gradient, axis=0)
    lp = K.learning_phase()
    saliency_function = K.function([model.input, y_test, lp], [saliency])
    return saliency_function


def get_saliency_gradient(loss_func, model, beta=1.0):
    shape = model.output_shape[1:]
    y_test = Input(shape=shape)
    y_pred = model.output
    gain_function = get_gain_function(loss_func, beta, y_test, y_pred)
    gradient = K.gradients(gain_function, model.input)[0]
    lp = K.learning_phase()
    saliency_gradient = K.function([model.input, y_test, lp], [gradient])
    return saliency_gradient


def get_gain_function(loss_func, beta, y_test, y_pred):
    if loss_func == 'categorical_crossentropy':
        y_pred_clipped = clip(K.epsilon(), 1.0 - K.epsilon())(y_pred)
        return -beta * K.sum(
            y_test * K.log(1. - y_pred_clipped) # + (1.0 - y_test) * K.square(y_pred)
            , axis=-1
        )
    elif loss_func == 'sklearn_hinge':
        y_pred_clipped = clip(-1. + K.epsilon(), 1. - K.epsilon())(y_pred)
        return beta * K.sum(y_test * K.square(K.relu(1. + y_pred_clipped)) +
                                 (1.0 - y_test) * K.square(K.relu(1. - y_pred_clipped)), axis=-1)
    elif loss_func == 'hinge':
        y_pred_clipped = clip(-1., 1.)(y_pred)
        return beta * K.sum(y_test * K.square(K.relu(1.0 + y_pred_clipped)) +
                                 (1.0 - y_test) * K.square(K.relu(1.0 - y_pred_clipped)), axis=-1)
    elif loss_func in ['mean_squared_error', 'mse']:
        # return y_pred * K.exp(-1.0 * K.square(y_test - y_pred))
        return 1.0 / (K.square(y_test - y_pred) + beta)
    else:
        raise Exception('Loss function ' + loss_func + ' not supported')


def get_factor(loss_func, y_test, y_pred):
    if loss_func == 'categorical_crossentropy':
        return K.sum(y_test * y_pred, axis=-1, keepdims=True)
    elif 'hinge' in loss_func:
        y_pred_clipped = K.stop_gradient(K.clip(y_pred, -1., 1.))
        y_pred_processed = .5 * (y_pred_clipped + 1.0)
        return K.sum(y_test * y_pred_processed, axis=-1, keepdims=True)
    elif loss_func == 'mean_squared_error':
        return 1.0 / (K.square(y_test - y_pred) + 1.0)
    else:
        raise Exception('Loss function ' + loss_func + ' not supported')
