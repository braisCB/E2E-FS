import os
import importlib


datasets = [
    'cifar100',
    'cifar10',
    'fashion_mnist',
    'mnist'
]

is_matlab = False


def main(rerun=False):
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../../')
    for dataset in datasets:
        print('RUNNING E2EFS MODELS FOR DATASET: ', dataset)
        if rerun or not os.path.isdir(os.path.dirname(os.path.realpath(__file__)) + '/' + dataset + '/info'):
            script_e2efs = importlib.import_module('scripts.deep.' + dataset + '.script')
            script_e2efs.main()


if __name__ == '__main__':
    main()
