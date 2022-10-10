import json
import numpy as np
import os
import glob


datasets = [
    'microarray/lymphoma',
    'microarray/colon',
    'microarray/leukemia',
    'microarray/lung181',
    'fs_challenge/dexter',
    'fs_challenge/gina',
    'fs_challenge/gisette',
]


def main(method):
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../')

    BA_AUCs = {}
    BA_10s = {}

    for dataset in datasets:
        directory = os.path.dirname(os.path.realpath(__file__)) + '/' + dataset + '/info/'
        files = glob.glob(directory + '*.json')

        for file in files:
            if method not in file.lower():
                continue
            print(file)
            with open(file, 'r') as outfile:
                stats = json.load(outfile)
            try:
                n_features = np.asarray(stats['classification']['n_features'])
                for key in ['auc', 'svc_auc', 'model_auc']:
                    if key not in stats['classification']:
                        continue
                    BA_key = key
                    if BA_key not in BA_AUCs:
                        BA_AUCs[BA_key] = ''
                        BA_10s[BA_key] = ''
                    BA = np.asarray(stats['classification'][key]).T
                    BA_AUC = (.5 * (BA[:, 1:] + BA[:, :-1]) * (n_features[1:] - n_features[:-1]) / (
                                n_features[-1] - n_features[0])).sum(axis=-1)
                    BA_AUCs[BA_key] += ' & ' + str(np.round(BA_AUC.mean(axis=0), 2)) + ' $\pm$ ' + str(np.round(BA_AUC.std(axis=0), 2))
                    BA_10s[BA_key] += ' & ' + str(np.round(BA[:,0].mean(axis=0), 2)) + ' $\pm$ ' + str(np.round(BA[:,0].std(axis=0), 2))
            except:
                for key in ['auc', 'svc_auc', 'model_auc']:
                    if key not in stats['classification']:
                        continue
                    BA_key = key
                    if BA_key not in BA_AUCs:
                        BA_AUCs[BA_key] = ''
                        BA_10s[BA_key] = ''
                    BA = np.asarray(stats['classification'][key]).T
                    BA_10s[BA_key] += ' & ' + str(np.round(BA.mean(), 2)) + ' $\pm$ ' + str(np.round(BA.std(), 2))

    for t, BA_dict in enumerate([BA_10s, BA_AUCs]):
        print('BA 10 features' if t == 0 else 'BA_AUC')

        keys = list(BA_dict.keys())
        for key in keys:
            print(key)
            print(BA_dict[key])


if __name__ == '__main__':
    method = 'naive'

    main(method)
