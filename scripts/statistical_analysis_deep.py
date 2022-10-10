import json
import numpy as np
import os
import glob
from scipy.stats import wilcoxon, friedmanchisquare
import scikit_posthocs as sp


def main(dataset, alpha=.05):
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../')

    directory = os.path.dirname(os.path.realpath(__file__)) + '/' + dataset + '/info/'
    files = glob.glob(directory + 'dense*.json')

    BA_AUCs = {}
    BA_10s = {}

    print(os.getcwd())
    for file in files:
        fs_class = file.split('.')[-2].split('_')[-4]
        with open(file, 'r') as outfile:
            stats = json.load(outfile)
        stats = stats[list(stats.keys())[0]][0]
        n_features = np.asarray(stats['classification']['n_features'])
        for key in ['accuracy', 'times']:
            if key not in stats['classification']:
                continue
            BA_key = fs_class + '_' + key
            BA = np.asarray(stats['classification'][key]).T
            print('method : ', fs_class)
            print(key, ' : ', BA.mean(axis=0), '+-', BA.std(axis=0))


    for t, BA_dict in enumerate([BA_10s, BA_AUCs]):
        print('BA 10 features' if t == 0 else 'BA_AUC')

        keys = list(BA_dict.keys())
        wilcoxon_matrix = np.zeros((len(keys), len(keys)))
        for i in range(len(keys) - 1):
            BA_i = BA_dict[keys[i]]
            for j in range(i+1, len(keys)):
                BA_j = BA_dict[keys[j]]
                t, p_value = wilcoxon(BA_i, BA_j)
                if p_value < alpha:
                    if BA_i.mean() > BA_j.mean():
                        wilcoxon_matrix[i, j] = 1
                        wilcoxon_matrix[j, i] = -1
                    else:
                        wilcoxon_matrix[i, j] = -1
                        wilcoxon_matrix[j, i] = 1

        # print(keys)
        # print(wilcoxon_matrix)

        min_wilkoxon = wilcoxon_matrix.min(axis=-1)
        max_wilkoxon = wilcoxon_matrix.max(axis=-1)
        best_methods = np.where((min_wilkoxon + 1) * max_wilkoxon > 0)[0]
        print('wilcoxon best methods : ', np.asarray(keys)[best_methods])


if __name__ == '__main__':
    dataset = 'deep/cifar10'
    alpha = .05

    main(dataset, alpha)
