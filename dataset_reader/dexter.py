import os
import numpy as np
from torchvision.datasets.utils import download_url
from scipy.special import erf

datasets_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


def download_dataset(directory):
    dataset_name = 'dexter'

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
    directory = (os.path.dirname(os.path.realpath(__file__)) + '/../datasets' if directory is None else directory) + '/dexter/'
    download_dataset(directory=directory)
    dataset = load_data(directory + 'dexter')
    return dataset


def load_data(source):
    info = {
        'train': {}, 'validation': {}, 'test': {}, 'raw': {}
    }

    file = source + '_train.labels'
    info['train']['label'] = np.loadtxt(file, dtype=np.int16)
    info['train']['label'][info['train']['label'] < 0] = 0

    file = source + '_train.data'
    info['train']['data'] = __load_dexter_data(file)

    file = source + '_test.data'
    info['test']['data'] = __load_dexter_data(file)

    file = source + '_valid.labels'
    info['validation']['label'] = np.loadtxt(file, dtype=np.int16)
    info['validation']['label'][info['validation']['label'] < 0] = 0

    file = source + '_valid.data'
    info['validation']['data'] = __load_dexter_data(file)

    info['raw']['data'] = np.concatenate((info['train']['data'], info['validation']['data']))
    info['raw']['label'] = np.concatenate((info['train']['label'], info['validation']['label']))

    return info

def __load_dexter_data(source):
    """
    A function that reads in the original dexter data in sparse form of feature:value
    and transform them into matrix form.
    # Arguments:
    filename: the url to either the dexter_train.data or dexter_valid.data
    mode: either 'text' for unpacked file; 'gz' for .gz file; or 'online' to download from the UCI repo
    # Return:
    the dexter data in matrix form.
    """
    with open(source) as f:
        readin_list = f.readlines()

    def to_dense_sparse(string_array):
        n = len(string_array)
        inds = np.zeros(n, dtype='int32')
        vals = np.zeros(n, dtype='int32')
        ret = np.zeros(20000, dtype='int32')
        for i in range(n):
            this_split = string_array[i].split(':')
            inds[i] = int(this_split[0]) - 1
            vals[i] = int(this_split[1])
        ret[inds] = vals
        return ret

    N = len(readin_list)
    dat = [None]*N

    for i in range(N):
        dat[i] = to_dense_sparse(readin_list[i].split(' ')[0:-1])[None, :]

    dat = np.concatenate(dat, axis=0).astype('float32')
    return dat


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
