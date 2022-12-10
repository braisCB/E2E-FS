from keras.utils import to_categorical
from keras import callbacks, optimizers as keras_optimizers
import json
import numpy as np
import os
from dataset_reader import gisette
from src.utils import balance_accuracy
from src.svc.models import LinearSVC
from extern.liblinear.python import liblinearutil
from src.baseline_methods import Fisher, ILFS, InfFS, MIM, ReliefF, SVMRFE, LASSORFE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from keras import backend as K
import time
import tensorflow as tf


fs_methods = [
    # Fisher.Fisher,
    ILFS.ILFS,
    InfFS.InfFS,
    MIM.MIM,
    ReliefF.ReliefF,
    SVMRFE.SVMRFE,
    LASSORFE.LASSORFE
]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


batch_size = 2
epochs = 150
mu = 100.
reps = 1
verbose = 0
k_folds = 3
k_fold_reps = 20
random_state = 42
optimizer_class = keras_optimizers.Adam
normalization_func = gisette.Normalize

dataset_name = 'gisette'
directory = os.path.dirname(os.path.realpath(__file__)) + '/info/'


initial_lr = .01


def scheduler():
    def sch(epoch):
        if epoch < 50:
            return initial_lr
        elif epoch < 100:
            return .2 * initial_lr
        else:
            return .04 * initial_lr

    return sch


def load_dataset():
    dataset = gisette.load_dataset()
    return dataset


def train_Keras(train_X, train_y, test_X, test_y, kwargs):
    normalization = normalization_func()
    num_classes = train_y.shape[-1]

    norm_train_X = normalization.fit_transform(train_X)
    norm_test_X = normalization.transform(test_X)

    class_weight = train_y.shape[0] / np.sum(train_y, axis=0)
    class_weight = num_classes * class_weight / class_weight.sum()
    sample_weight = None
    batch_size = max(2, len(norm_train_X) // 50)
    print('reps : ', reps, ', weights : ', class_weight)
    if num_classes == 2:
        sample_weight = np.zeros((len(norm_train_X),))
        sample_weight[train_y[:, 1] == 1] = class_weight[1]
        sample_weight[train_y[:, 1] == 0] = class_weight[0]
        class_weight = None

    svc_model = LinearSVC(nfeatures=norm_train_X.shape[1:], **kwargs)
    svc_model.create_keras_model(nclasses=num_classes, warming_up=False)
    model = svc_model.model

    optimizer = optimizer_class(lr=initial_lr)

    model.compile(
        loss=LinearSVC.loss_function('square_hinge', class_weight),
        optimizer=optimizer,
        metrics=[LinearSVC.accuracy]
    )

    model.fit(
        norm_train_X, train_y, batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            callbacks.LearningRateScheduler(scheduler()),
        ],
        validation_data=(norm_test_X, test_y),
        class_weight=class_weight,
        sample_weight=sample_weight,
        verbose=verbose
    )

    model.normalization = normalization

    return model


def train_SVC(train_X, train_y, kwargs):
    params = '-q -s ' + str(kwargs['solver']) + ' -c ' + str(kwargs['C'])
    model = liblinearutil.train((2 * train_y[:, -1] - 1).tolist(), train_X.tolist(), params)
    return model


def main(dataset_name):

    dataset = load_dataset()

    raw_data = np.asarray(dataset['raw']['data'])
    raw_label = np.asarray(dataset['raw']['label'])
    num_classes = len(np.unique(raw_label))

    rskf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=k_fold_reps, random_state=random_state)

    for fs_method in fs_methods:
        print('FS-Method : ', fs_method.__name__)
        cont_seed = 0

        nfeats = []
        accuracies = []
        svc_accuracies = []
        BAs = []
        svc_BAs = []
        fs_time = []
        aucs = []
        svc_aucs = []
        mus = []
        name = dataset_name + '_mu_' + str(mu)
        print(name, 'samples : ', raw_label.sum(), (1. - raw_label).sum())

        for j, (train_index, test_index) in enumerate(rskf.split(raw_data, raw_label)):
            print('k_fold', j, 'of', k_folds*k_fold_reps)

            train_data, train_labels = raw_data[train_index].copy(), raw_label[train_index].copy()
            test_data, test_labels = raw_data[test_index].copy(), raw_label[test_index].copy()

            train_labels = to_categorical(train_labels, num_classes=num_classes)
            test_labels = to_categorical(test_labels, num_classes=num_classes)

            valid_features = np.where(np.abs(train_data).sum(axis=0) > 0)[0]
            if len(valid_features) < train_data.shape[1]:
                print('Removing', train_data.shape[1] - len(valid_features), 'zero features')
                train_data = train_data[:, valid_features]
                test_data = test_data[:, valid_features]

            model_kwargs = {
                'mu': mu / len(train_data),
                'degree': 3
            }
            print('mu :', model_kwargs['mu'], ', batch_size :', batch_size)

            svc_kwargs = {
                'C': 1.0,
                'solver': 0.
            }

            print('Starting feature selection')
            fs_dir = os.path.dirname(os.path.realpath(__file__)) + '/temp/'
            if not os.path.isdir(fs_dir):
                os.makedirs(fs_dir)
            fs_filename = fs_dir + fs_method.__name__ + '_iter_' + str(j) + '_seed_' + \
                          str(random_state) + '.json'
            if os.path.exists(fs_filename):
                with open(fs_filename, 'r') as outfile:
                    fs_data = json.load(outfile)
                fs_class = fs_method(n_features_to_select=200 if 'RFE' not in fs_method.__name__ else 10)
                fs_class.score = np.asarray(fs_data['score'])
                fs_class.ranking = np.asarray(fs_data['ranking'])
                fs_time.append(np.NAN)
            else:
                start_time = time.process_time()
                fs_class = fs_method(n_features_to_select=200 if 'RFE' not in fs_method.__name__ else 10)
                fs_class.fit(train_data, 2. * train_labels[:, -1] - 1.)
                fs_data = {
                    'score': fs_class.score.tolist(),
                    'ranking': fs_class.ranking.tolist()
                }
                fs_time.append(time.process_time() - start_time)
                with open(fs_filename, 'w') as outfile:
                    json.dump(fs_data, outfile)
            print('Finishing feature selection. Time : ', fs_time[-1], 's')

            for i, n_features in enumerate([10, 50, 100, 150, 200]):
                n_accuracies = []
                n_svc_accuracies = []
                n_BAs = []
                n_svc_BAs = []
                n_aucs = []
                n_svc_aucs = []
                n_train_accuracies = []
                print('n_features : ', n_features)

                fs_class.n_features_to_select = n_features
                svc_train_data = fs_class.transform(train_data)
                svc_test_data = fs_class.transform(test_data)

                norm = normalization_func()
                svc_train_data_norm = norm.fit_transform(svc_train_data)
                svc_test_data_norm = norm.transform(svc_test_data)

                bestcv = -1
                bestc = None
                bestSolver = None
                for s in [0, 1, 2, 3]:
                    for my_c in [0.001, 0.01, 0.1, 0.5, 1.0, 1.4, 1.5, 1.6, 2.0, 2.5, 5.0, 25.0, 50.0, 100.0]:
                        cmd = '-v 5 -s ' + str(s) + ' -c ' + str(my_c) + ' -q'
                        cv = liblinearutil.train((2 * train_labels[:, -1] - 1).tolist(), svc_train_data_norm.tolist(), cmd)
                        if cv > bestcv:
                            bestcv = cv
                            bestc = my_c
                            bestSolver = s
                svc_kwargs['C'] = bestc
                svc_kwargs['solver'] = bestSolver
                print('Best -> C:', bestc, ', s:', bestSolver, ', acc:', bestcv)

                for r in range(reps):
                    np.random.seed(cont_seed)
                    tf.random.set_seed(cont_seed)
                    cont_seed += 1

                    model = train_SVC(svc_train_data_norm, train_labels, svc_kwargs)
                    _, accuracy, test_pred = liblinearutil.predict(
                        (2 * test_labels[:, -1] - 1).tolist(), svc_test_data_norm.tolist(), model, '-q'
                    )
                    test_pred = np.asarray(test_pred)
                    n_svc_accuracies.append(accuracy[0])
                    n_svc_BAs.append(balance_accuracy(test_labels, test_pred))
                    n_svc_aucs.append(roc_auc_score(test_labels[:, -1], test_pred))
                    del model
                    model = train_Keras(svc_train_data, train_labels, svc_test_data, test_labels, model_kwargs)
                    train_data_norm = model.normalization.transform(svc_train_data)
                    test_data_norm = model.normalization.transform(svc_test_data)
                    test_pred = model.predict(test_data_norm)
                    n_BAs.append(balance_accuracy(test_labels, test_pred))
                    n_aucs.append(roc_auc_score(test_labels[:, -1], test_pred))
                    n_accuracies.append(model.evaluate(test_data_norm, test_labels, verbose=0)[-1])
                    n_train_accuracies.append(model.evaluate(train_data_norm, train_labels, verbose=0)[-1])
                    del model
                    K.clear_session()
                    print(
                        'n_features : ', n_features,
                        ', acc : ', n_accuracies[-1],
                        ', BA : ', n_BAs[-1],
                        ', auc : ', n_aucs[-1],
                        ', train_acc : ', n_train_accuracies[-1],
                        ', svc_acc : ', n_svc_accuracies[-1],
                        ', svc_BA : ', n_svc_BAs[-1],
                        ', svc_auc : ', n_svc_aucs[-1],
                    )
                if i >= len(accuracies):
                    accuracies.append(n_accuracies)
                    svc_accuracies.append(n_svc_accuracies)
                    BAs.append(n_BAs)
                    aucs.append(n_aucs)
                    svc_BAs.append(n_svc_BAs)
                    svc_aucs.append(n_svc_aucs)
                    nfeats.append(n_features)
                    mus.append(model_kwargs['mu'])
                else:
                    accuracies[i] += n_accuracies
                    svc_accuracies[i] += n_svc_accuracies
                    BAs[i] += n_BAs
                    aucs[i] += n_aucs
                    svc_BAs[i] += n_svc_BAs
                    svc_aucs[i] += n_svc_aucs

        mean_accuracies = np.array(accuracies).mean(axis=-1)
        print('NFEATS : ', nfeats)
        diff_accuracies = .5 * (mean_accuracies[1:] + mean_accuracies[:-1])
        np_nfeats = np.array(nfeats)
        diff = np_nfeats[1:] - np_nfeats[:-1]
        AUC = np.sum(diff * diff_accuracies) / np.sum(diff)
        print('AUC : ', AUC)

        output_filename = directory + 'LinearSVC_' + fs_method.__name__ + '.json'

        if not os.path.isdir(directory):
            os.makedirs(directory)

        info_data = {
            'reps': reps,
            'classification': {
                'mus': mus,
                'n_features': nfeats,
                'accuracy': accuracies,
                'mean_accuracy': np.array(accuracies).mean(axis=1).tolist(),
                'svc_accuracy': svc_accuracies,
                'mean_svc_accuracy': np.array(svc_accuracies).mean(axis=1).tolist(),
                'BA': BAs,
                'mean_BA': np.array(BAs).mean(axis=1).tolist(),
                'auc': aucs,
                'mean_auc': np.array(aucs).mean(axis=1).tolist(),
                'svc_BA': svc_BAs,
                'svc_mean_BA': np.array(svc_BAs).mean(axis=1).tolist(),
                'svc_auc': svc_aucs,
                'svc_mean_auc': np.array(svc_aucs).mean(axis=1).tolist(),
                'fs_time': fs_time
            }
        }

        for k, v in info_data['classification'].items():
            if 'mean' in k:
                print(k, v)

        with open(output_filename, 'w') as outfile:
            json.dump(info_data, outfile)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../../../')
    main(dataset_name)