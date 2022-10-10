import json
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import seaborn as sns


def main(dataset):
    dataset_name = dataset.split('/')[-1]
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../')

    directory = os.path.dirname(os.path.realpath(__file__)) + '/' + dataset + '/info/'
    image_directory = os.path.dirname(os.path.realpath(__file__)) + '/images_kde/'
    files = glob.glob(directory + '*.json')

    BA_means = {}

    print(os.getcwd())
    for file in files:
        fs_class = file.split('.')[-2].split('_')[-1]
        with open(file, 'r') as outfile:
            stats = json.load(outfile)
        # n_features = np.asarray(stats['classification']['n_features'])
        for key in ['BA']:
            if key not in stats['classification']:
                continue
            BA = np.asarray(stats['classification'][key])
            if BA.ndim == 2 and BA.shape[1] == 1:
                BA = BA.flatten()
            BA_mean = BA[0]  # .mean(axis=0)
            if isinstance(BA_mean, float):
                BA_mean = np.asarray([BA_mean] * 5)
            BA_means[fs_class] = BA_mean
            print('method : ', fs_class)
            print('score', key, ' : ', BA_mean)

    keys = list(sorted(BA_means.keys()))
    # keys = ['SFS', 'E2EFS', 'Fisher', 'DFS']
    if not os.path.isdir(image_directory):
        os.makedirs(image_directory)
    values = []
    for i, key in enumerate(keys):
        axes = sns.kdeplot(data=BA_means[key])
        axes.set_title(dataset_name)
        axes.legend(keys[:i+1], loc='best', fontsize=12)
        fig = axes.get_figure()
        fig.savefig(image_directory + dataset_name + '_' + key + '.png')
        values.append(BA_means[key])

    print(keys)
    print(np.argsort(-np.array(values), axis=0).mean(axis=1))


if __name__ == '__main__':
    dataset = 'microarray/colon'
    main(dataset)

