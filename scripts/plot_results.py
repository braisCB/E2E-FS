import json
import numpy as np
import os
import glob
from matplotlib import pyplot as plt


def main(dataset):
    dataset_name = dataset.split('/')[-1]
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../')

    directory = os.path.dirname(os.path.realpath(__file__)) + '/' + dataset + '/info/'
    image_directory = os.path.dirname(os.path.realpath(__file__)) + '/images/'
    files = glob.glob(directory + '*.json')

    BA_means = {}

    print(os.getcwd())
    for file in files:
        fs_class = file.split('.')[-2].split('_')[-1]
        with open(file, 'r') as outfile:
            stats = json.load(outfile)
        n_features = np.asarray(stats['classification']['n_features'])
        for key in ['fs_time']:
            if key not in stats['classification']:
                continue
            BA = np.asarray(stats['classification'][key])
            if BA.ndim == 2 and BA.shape[1] == 1:
                BA = BA.flatten()
            BA_mean = BA.mean(axis=-1)
            if isinstance(BA_mean, float):
                BA_mean = np.asarray([BA_mean] * 5)
            BA_means[fs_class] = BA_mean
            print('method : ', fs_class)
            print('score', key, ' : ', BA_mean)

    keys = list(sorted(BA_means.keys()))
    markers = ['o', 'v', '^', 'p', '*', '+', 'x', '<', '>']
    plt.figure()
    for i, key in enumerate(keys):
        plt.plot(n_features, BA_means[key], marker=markers[i % len(markers)], linewidth=2)
    plt.legend(keys, loc='best', fontsize=12)
    plt.title(dataset_name.upper(), fontsize=14)
    plt.xlabel('# features')
    plt.ylabel('time (s)')
    if not os.path.isdir(image_directory):
        os.makedirs(image_directory)
    plt.savefig(image_directory + dataset_name + '_fs_time.png')
    plt.show()


if __name__ == '__main__':
    dataset = 'fs_challenge/madelon'
    main(dataset)

