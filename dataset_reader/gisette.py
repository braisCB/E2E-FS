import os
import numpy as np
from torchvision.datasets.utils import download_url
from scipy.special import erf

datasets_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


def download_dataset(directory):
    dataset_name = 'gisette'

    if not os.path.isdir(directory):
        os.makedirs(directory)

    train_url = datasets_url + dataset_name.lower() + '/'

    for subset in ['train', 'test', 'valid']:
        for option in ['labels', 'data']:
            if option == 'labels' and subset == 'test':
                continue
            filename = dataset_name + '_' + subset + '.' + option
            if subset == 'valid' and option == 'labels':
                url = train_url + dataset_name.lower() + '_' + subset + '.' + option
            else:
                url = train_url + dataset_name.upper() + '/' + dataset_name.lower() + '_' + subset + '.' + option
            if not os.path.exists(directory + '/' + filename):
                download_url(url=url, filename=filename, root=directory)


def load_dataset(directory=None):
    directory = (os.path.dirname(os.path.realpath(__file__)) + '/../datasets' if directory is None else directory) + '/gisette/'
    download_dataset(directory=directory)
    dataset = load_data(directory + 'gisette')
    return dataset


def load_data(source):
    info = {
        'train': {}, 'validation': {}, 'test': {}, 'raw': {}
    }

    file = source + '_train.labels'
    info['train']['label'] = np.loadtxt(file, dtype=np.int16)
    info['train']['label'][info['train']['label'] < 0] = 0

    file = source + '_train.data'
    info['train']['data'] = np.loadtxt(file, dtype=np.int16).astype(np.float32)

    file = source + '_test.data'
    info['test']['data'] = np.loadtxt(file, dtype=np.int16).astype(np.float32)

    file = source + '_valid.labels'
    info['validation']['label'] = np.loadtxt(file, dtype=np.int16)
    info['validation']['label'][info['validation']['label'] < 0] = 0

    file = source + '_valid.data'
    info['validation']['data'] = np.loadtxt(file, dtype=np.int16).astype(np.float32)

    info['raw']['data'] = np.concatenate((info['train']['data'], info['validation']['data']))
    info['raw']['label'] = np.concatenate((info['train']['label'], info['validation']['label']))

    return info


class Normalize:

    def __init__(self):
        self.stats = None

    def fit(self, X):
        mean = np.mean(X, axis=0)
        std = np.sqrt(np.square(X - mean).sum(axis=0) / max(1, len(X) - 1))
        self.stats = (mean, std)

    def transform(self, X):
        transformed_X = erf((X - self.stats[0]) / (np.maximum(1e-6, self.stats[1]) * np.sqrt(2.)))
        return transformed_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
