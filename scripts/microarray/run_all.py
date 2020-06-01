import os
import importlib


datasets = [
    'colon',
    'lymphoma',
    'leukemia',
    'lung181'
]

is_matlab = False


def main(rerun=False):
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../../')
    for dataset in datasets:
        print('RUNNING E2EFS MODELS FOR DATASET: ', dataset)
        if not rerun and os.path.isdir(os.path.dirname(os.path.realpath(__file__)) + '/' + dataset + '/info'):
            continue
        script_e2efs = importlib.import_module('scripts.microarray.' + dataset + '.script_e2efs')
        script_e2efs.main(dataset)

        print('RUNNING BASELINE MODELS FOR DATASET: ', dataset)
        script_baseline = importlib.import_module('scripts.microarray.' + dataset + '.script_baseline')
        script_baseline.main(dataset)
        if is_matlab:
            print('RUNNING MATLAB BASELINE MODELS FOR DATASET: ', dataset)
            script_baseline_matlab = importlib.import_module('scripts.microarray.' + dataset + '.script_baseline_matlab')
            script_baseline_matlab.main(dataset)
    statistical_analysis = importlib.import_module('scripts.statistical_analysis')
    for dataset in datasets:
        statistical_analysis.main('microarray/' + dataset)


if __name__ == '__main__':
    main()
