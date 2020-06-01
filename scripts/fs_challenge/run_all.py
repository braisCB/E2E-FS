import os
import importlib


datasets = [
    # 'gina',
    # 'dexter',
    # 'gisette',
    'madelon'
]

is_matlab = False


def main(rerun=False):
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../../')
    for dataset in datasets:
        print('RUNNING E2EFS MODELS FOR DATASET: ', dataset)
        if dataset != 'madelon' and (rerun or not os.path.isdir(os.path.dirname(os.path.realpath(__file__)) + '/' + dataset + '/info')):
            script_e2efs = importlib.import_module('scripts.fs_challenge.' + dataset + '.script_e2efs')
            script_e2efs.main(dataset)
            script_baseline = importlib.import_module('scripts.fs_challenge.' + dataset + '.script_baseline')
            script_baseline.main(dataset)
            if is_matlab:
                print('RUNNING MATLAB BASELINE MODELS FOR DATASET: ', dataset)
                script_baseline_matlab = importlib.import_module(
                    'scripts.fs_challenge.' + dataset + '.script_baseline_matlab')
                script_baseline_matlab.main(dataset)
        if rerun or not os.path.isdir(os.path.dirname(os.path.realpath(__file__)) + '/' + dataset + '/info_nn'):
            script_e2efs_nn = importlib.import_module('scripts.fs_challenge.' + dataset + '.script_e2efs_nn')
            script_e2efs_nn.main(dataset)
            script_baseline_nn = importlib.import_module('scripts.fs_challenge.' + dataset + '.script_baseline_nn')
            script_baseline_nn.main(dataset)
    statistical_analysis = importlib.import_module('scripts.statistical_analysis')
    plot_results = importlib.import_module('scripts.plot_results')
    for dataset in datasets:
        if dataset != 'madelon':
            statistical_analysis.main('fs_challenge/' + dataset)
        plot_results.main('fs_challenge/' + dataset)


if __name__ == '__main__':
    main()
