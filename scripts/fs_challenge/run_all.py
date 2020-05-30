import os
import importlib


datasets = [
    'gina',
    'dexter',
    'gisette'
]

is_matlab = False


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../../')
    for dataset in datasets:
        print('RUNNING E2EFS MODELS FOR DATASET: ', dataset)
        script_e2efs = importlib.import_module('scripts.fs_challenge.' + dataset + '.script_e2efs')
        script_e2efs.main(dataset)
        script_e2efs_nn = importlib.import_module('scripts.fs_challenge.' + dataset + '.script_e2efs_nn')
        script_e2efs_nn.main(dataset)

        print('RUNNING BASELINE MODELS FOR DATASET: ', dataset)
        script_baseline = importlib.import_module('scripts.fs_challenge.' + dataset + '.script_baseline')
        script_baseline.main(dataset)
        script_baseline_nn = importlib.import_module('scripts.fs_challenge.' + dataset + '.script_baseline_nn')
        script_baseline_nn.main(dataset)
        if is_matlab:
            print('RUNNING MATLAB BASELINE MODELS FOR DATASET: ', dataset)
            script_baseline_matlab = importlib.import_module('scripts.fs_challenge.' + dataset + '.script_baseline_matlab')
            script_baseline_matlab.main(dataset)


if __name__ == '__main__':
    main()
