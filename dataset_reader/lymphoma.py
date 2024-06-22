import numpy as np
from scipy.special import erf
from scipy.io import loadmat
import os


data_filename = 'lymphoma.mat'

def load_dataset(directory=None):
    directory = (os.path.dirname(os.path.realpath(__file__)) + '/../datasets' if directory is None else directory) + '/lymphoma/'
    dataset = load_data(directory)
    return dataset


def load_data(directory):
    info = {
        'raw': {}
    }

    mat = loadmat(directory + data_filename)

    with open(directory + data_filename) as f:
        info['raw']['data'] = np.asarray(mat['data']).T
        info['raw']['label'] = np.asarray(mat['labels']).astype(int)
        info['raw']['label'][info['raw']['label'] == 2] = 0

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
