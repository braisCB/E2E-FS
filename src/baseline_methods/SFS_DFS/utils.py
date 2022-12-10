import numpy as np
from keras.utils import to_categorical
from copy import deepcopy


def dict_merge(a, b):
    "merges b into a"
    output = deepcopy(a)
    b = {} if b is None else b
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                output[key] = dict_merge(a[key], b[key])
            else:
                output[key] = b[key] # same leaf value
        else:
            output[key] = b[key]
    return output


def balance_data(data, data_label):
    label = data_label if data_label.ndim == 1 else np.argmax(data_label, axis=1)
    uclasses = np.unique(label)
    nclasses = len(uclasses)

    freq = np.zeros(nclasses).astype(int)
    for i, cl in enumerate(uclasses):
        freq[i] = (label == cl).sum()

    n_samples = np.max(freq)
    new_data = []
    new_label = []
    for cl in uclasses:
        data_cl = data[label == cl]
        label_cl = label[label == cl]
        n_samples_cl = len(label_cl)
        index = 0
        while index < n_samples:
            perm = np.random.permutation(n_samples_cl)
            end_index = min(n_samples_cl, n_samples - index)
            new_data.append(data_cl[perm[:end_index]])
            new_label.append(label_cl[perm[:end_index]])
            index += end_index

    new_data, new_label = np.concatenate(new_data, axis=0), np.concatenate(new_label, axis=0)
    new_label = new_label if data_label.ndim == 1 else to_categorical(new_label, nclasses)

    return new_data, new_label
