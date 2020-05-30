import numpy as np
from keras.utils import to_categorical


def balance_accuracy(y_true, y_pred):
    y_pred[np.isnan(y_pred)] = 0.
    y_pred[np.isinf(y_pred)] = 0.
    if len(y_pred.shape) == 1 or y_pred.shape[-1] == 1:
        y_pred_index = np.maximum(0., np.sign(y_pred)).astype(int)
    else:
        y_pred_index = np.argmax(y_pred, axis=-1)
    if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
        y_true_index = np.maximum(0., np.sign(y_true)).astype(int)
    else:
        y_true_index = np.argmax(y_true, axis=-1)
    y_pred_one_hot = to_categorical(y_pred_index, 2)
    y_true_one_hot = to_categorical(y_true_index, 2)
    return float(np.mean(np.sum(y_true_one_hot * y_pred_one_hot, axis=0) / np.maximum(1., np.sum(y_true_one_hot, axis=0))))
