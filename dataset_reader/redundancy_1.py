import numpy as np
from scipy.special import erf
import os
import pickle


data_filename = 'redundancy_1.pickle'
REAL_FEATURES = 5
REDUNDANT_FEATURES = 15
RANDOM_FEATURES = 480
N_SAMPLES = 2500


def create_dataset(directory=None, seed=46):
    directory = (os.path.dirname(os.path.realpath(__file__)) + '/../datasets' if directory is None else directory) + '/redundancy_1/'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    np.random.seed(seed)
    real_features = np.random.randn(N_SAMPLES, REAL_FEATURES)
    redundant_features = real_features @ np.random.randn(REAL_FEATURES, REDUNDANT_FEATURES)
    random_features = np.random.randn(N_SAMPLES, RANDOM_FEATURES)
    data = np.concatenate((real_features, redundant_features, random_features), axis=1)
    labels = np.sign(real_features.sum(axis=1, keepdims=True))
    with open(directory + data_filename, 'wb') as handle:
        pickle.dump({'data': data, 'labels': labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset(directory=None):
    directory = (os.path.dirname(os.path.realpath(__file__)) + '/../datasets' if directory is None else directory) + '/redundancy_1/'
    dataset = load_data(directory)
    return dataset


def load_data(directory):
    info = {
        'raw': {}
    }

    with open(directory + data_filename, 'rb') as handle:
        mat = pickle.load(handle)

    with open(directory + data_filename) as f:
        info['raw']['data'] = np.asarray(mat['data'])
        info['raw']['label'] = np.asarray(mat['labels']).astype(int)
        info['raw']['label'][info['raw']['label'] == -1] = 0

    return info


class Normalize:

    def __init__(self):
        self.stats = None

    def fit(self, X):
        X_mean = np.mean(X, axis=0)
        X_std = np.sqrt(np.square(X - X_mean).sum(axis=0) / max(1, len(X) - 1))
        self.stats = (X_mean, X_std)

    def transform(self, X):
        transformed_X = erf((X - self.stats[0]) / (np.maximum(1e-6, self.stats[1]) * np.sqrt(2.)))
        return transformed_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
    create_dataset()
