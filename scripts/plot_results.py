import json
import numpy as np
import os
import glob
from matplotlib import pyplot as plt


def main(dataset):
    dataset_name = dataset.split('/')[-1]
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../')

    directory = os.path.dirname(os.path.realpath(__file__)) + '/' + dataset + '/info_nn/'
    files = glob.glob(directory + '*.json')

    BA_means = {}

    print(os.getcwd())
    for file in files:
        fs_class = file.split('.')[-2].split('_')[-1]
        with open(file, 'r') as outfile:
            stats = json.load(outfile)
        n_features = np.asarray(stats['classification']['n_features'])
        for key in ['BA']:
            if key not in stats['classification']:
                continue
            BA = np.asarray(stats['classification'][key])
            BA_mean = BA.mean(axis=-1)
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
    plt.savefig(directory + dataset_name + '_nn_results.png')
    plt.show()


if __name__ == '__main__':
    dataset = 'fs_challenge/madelon'
    main(dataset)

