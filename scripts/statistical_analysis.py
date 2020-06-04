import json
import numpy as np
import os
import glob
from scipy.stats import wilcoxon


def main(dataset, alpha=.05):
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../')

    directory = os.path.dirname(os.path.realpath(__file__)) + '/' + dataset + '/info/'
    files = glob.glob(directory + '*.json')

    BA_AUCs = {}

    print(os.getcwd())
    for file in files:
        fs_class = file.split('.')[-2].split('_')[-1]
        with open(file, 'r') as outfile:
            stats = json.load(outfile)
        n_features = np.asarray(stats['classification']['n_features'])
        for key in ['BA', 'svc_BA', 'model_BA']:
            if key not in stats['classification']:
                continue
            BA_key = fs_class + '_' + key.split('_')
            BA = np.asarray(stats['classification'][key]).T
            BA_AUC = (.5 * (BA[:, 1:] + BA[:, :-1]) / (n_features[1:] - n_features[:-1])).sum(axis=-1)
            BA_AUCs[BA_key] = BA_AUC
            print('method : ', fs_class)
            print('score', key, ' : ', BA.mean(), '+-', BA.std())

    keys = list(BA_AUCs.keys())
    wilcoxon_matrix = np.zeros((len(keys), len(keys)))
    for i in range(len(keys) - 1):
        BA_i = BA_AUCs[keys[i]]
        for j in range(i+1, len(keys)):
            BA_j = BA_AUCs[keys[j]]
            t, p_value = wilcoxon(BA_i, BA_j)
            if p_value < alpha:
                if BA_i.mean() > BA_j.mean():
                    wilcoxon_matrix[i, j] = 1
                    wilcoxon_matrix[j, i] = -1
                else:
                    wilcoxon_matrix[i, j] = -1
                    wilcoxon_matrix[j, i] = 1

    print(keys)
    print(wilcoxon_matrix)

    min_wilkoxon = wilcoxon_matrix.min(axis=-1)
    max_wilkoxon = wilcoxon_matrix.max(axis=-1)
    best_methods = np.where((min_wilkoxon + 1) * max_wilkoxon > 0)[0]
    print('best methods : ', np.asarray(keys)[best_methods])

if __name__ == '__main__':
    dataset = 'microarray/colon'
    alpha = .05

    main(dataset, alpha)
