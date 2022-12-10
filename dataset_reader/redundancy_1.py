import numpy as np
from scipy.special import erf
import os
import pickle
from keras.utils import to_categorical


data_filename = 'redundancy_1.pickle'
mask_filename = 'redundancy_1_mask.pickle'
REAL_FEATURES = 20
REDUNDANT_FEATURES = 80
RANDOM_FEATURES = 400
N_SAMPLES = 2500
RANDOM_STD = .05
N_CLASSES = 10


def create_dataset(directory=None, seed=46):
    directory = (os.path.dirname(os.path.realpath(__file__)) + '/../datasets' if directory is None else directory) + '/redundancy_1/'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    np.random.seed(seed)
    factor = np.random.randn(REAL_FEATURES, 1)
    bias = np.random.randn(N_CLASSES, REAL_FEATURES)
    real_features = np.random.randn(N_SAMPLES, REAL_FEATURES)
    real_mask = to_categorical(np.arange(REAL_FEATURES))
    redundancy_mask = to_categorical(np.argmax(np.random.randn(REDUNDANT_FEATURES, REAL_FEATURES), axis=1), REAL_FEATURES)
    random_mask = np.zeros((N_SAMPLES, REAL_FEATURES))
    redundant_features = real_features @ redundancy_mask.T
    random_features = np.random.randn(N_SAMPLES, RANDOM_FEATURES)
    data = np.concatenate((real_features, redundant_features, random_features), axis=1)
    data += RANDOM_STD * np.random.randn(*data.shape)
    mask = np.concatenate((real_mask, redundancy_mask, random_mask), axis=0)
    dist = np.sqrt(np.sum(np.square(real_features[:, np.newaxis, :] - bias), axis=2))
    labels = np.argmax(dist, axis=1)[:, None]
    with open(directory + data_filename, 'wb') as handle:
        pickle.dump({'data': data, 'labels': labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(directory + mask_filename, 'wb') as handle:
        pickle.dump({'mask': mask, 'factor': factor}, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


def get_redundancy_stats(feats, directory=None):
    directory = (os.path.dirname(os.path.realpath(__file__)) + '/../datasets' if directory is None else directory) + '/redundancy_1/'
    with open(directory + mask_filename, 'rb') as handle:
        mask = pickle.load(handle)['mask']
    feat_mask = mask[feats].sum(axis=0)
    real_feats = np.sign(feat_mask).sum()
    redundant_feats = feat_mask.sum() - real_feats
    return real_feats/float(REAL_FEATURES), redundant_feats/float(REDUNDANT_FEATURES)


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
