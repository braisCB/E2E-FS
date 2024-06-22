import os
import numpy as np
from torchvision.datasets.utils import download_url
import zipfile
import pandas as pd
from scipy.special import erf


gina_agnostic = 'http://www.agnostic.inf.ethz.ch/datasets/DataAgnos/GINA.zip'
gina_valid_labels = 'http://www.agnostic.inf.ethz.ch/datasets/ValidAgnos.zip'
gina_agnostic_filename = 'gina_agnostic.zip'
gina_valid_labels_filename = 'gina_valid_labels.zip'

def download_dataset(directory=None):

    directory = (os.path.dirname(os.path.realpath(__file__)) + '/../datasets' if directory is None else directory) + '/gina/'

    if not os.path.isdir(directory):
        os.makedirs(directory)

    if not os.path.exists(directory + gina_agnostic_filename):
        download_url(url=gina_agnostic, filename=gina_agnostic_filename, root=directory)

    if not os.path.exists(directory + gina_valid_labels_filename):
        download_url(url=gina_valid_labels, filename=gina_valid_labels_filename, root=directory)


def load_dataset(directory=None):
    download_dataset(directory=directory)
    directory = (os.path.dirname(os.path.realpath(__file__)) + '/../datasets' if directory is None else directory) + '/gina/'
    dataset = load_data(directory)
    return dataset


def load_data(directory):
    info = {
        'train': {}, 'validation': {}, 'test': {}, 'raw': {}
    }

    gina_archive = directory + gina_agnostic_filename
    archive = zipfile.ZipFile(gina_archive, 'r')

    test_data_file = archive.filelist[1].filename
    train_data_file = archive.filelist[2].filename
    train_label_file = archive.filelist[3].filename
    valid_data_file = archive.filelist[4].filename

    with archive.open(test_data_file) as f:
        info['test']['data'] = pd.read_csv(f, sep=' ').values[:,:-1]

    with archive.open(train_data_file) as f:
        info['train']['data'] = pd.read_csv(f, sep=' ').values[:,:-1]

    with archive.open(train_label_file) as f:
        info['train']['label'] = pd.read_csv(f).values

    with archive.open(valid_data_file) as f:
        info['validation']['data'] = pd.read_csv(f, sep=' ').values[:,:-1]

    gina_valid_label_archive = directory + gina_valid_labels_filename
    archive = zipfile.ZipFile(gina_valid_label_archive, 'r')

    valid_label_file = archive.filelist[1].filename

    with archive.open(valid_label_file) as f:
        info['validation']['label'] = pd.read_csv(f).values

    info['train']['label'] = np.maximum(0., info['train']['label'])
    info['validation']['label'] = np.maximum(0., info['validation']['label'])

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

