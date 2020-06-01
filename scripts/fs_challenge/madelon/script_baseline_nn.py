from keras.utils import to_categorical
from keras import callbacks, optimizers as keras_optimizers
import json
import numpy as np
import os
from dataset_reader import madelon
from src.utils import balance_accuracy
from src.network_models import three_layer_nn
from src.baseline_methods import Fisher, ILFS, InfFS, MIM, ReliefF
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score
from keras import backend as K


fs_methods = [
    ILFS.ILFS,
    Fisher.Fisher,
    InfFS.InfFS,
    MIM.MIM,
    ReliefF.ReliefF,
]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 2
epochs = 150
reps = 1
verbose = 0
k_folds = 3
k_fold_reps = 20
regularization = 1e-2
random_state = 42
optimizer_class = keras_optimizers.Adam
normalization_func = madelon.Normalize

dataset_name = 'madelon'
directory = os.path.dirname(os.path.realpath(__file__)) + '/info_nn/'


def scheduler(extra=0, factor=.1):
    def sch(epoch):
        if epoch < 50 + extra:
            return .01 * factor
        elif epoch < 100 + extra:
            return .002 * factor
        else:
            return .0004 * factor

    return sch


def load_dataset():
    dataset = madelon.load_dataset()
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

    model = three_layer_nn(nfeatures=norm_train_X.shape[1:], **kwargs)
    optimizer = optimizer_class(lr=1e-3)  # optimizers.adam(lr=1e-2)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['acc']
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


def main(dataset_name):

    dataset = load_dataset()

    raw_data = np.asarray(dataset['raw']['data'])
    raw_label = np.asarray(dataset['raw']['label'])
    num_classes = len(np.unique(raw_label))

    rskf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=k_fold_reps, random_state=random_state)

    for fs_method in fs_methods:
        print('FS-Method : ', fs_method.__name__)

        nfeats = []
        accuracies = []
        BAs = []
        mAPs = []
        name = dataset_name + '_three_layer_nn'
        print(name)

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
                'regularization': regularization,
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
                fs_class = fs_method(n_features_to_select=200)
                fs_class.score = np.asarray(fs_data['score'])
                fs_class.ranking = np.asarray(fs_data['ranking'])
            else:
                fs_class = fs_method(n_features_to_select=200)
                fs_class.fit(train_data, 2. * train_labels[:, -1] - 1.)
                fs_data = {
                    'score' : fs_class.score.tolist(),
                    'ranking': fs_class.ranking.tolist()
                }
                with open(fs_filename, 'w') as outfile:
                    json.dump(fs_data, outfile)

            print('Finishing feature selection')

            for i, n_features in enumerate([5, 10, 15, 20]):
                n_accuracies = []
                n_BAs = []
                n_mAPs = []
                n_train_accuracies = []
                print('n_features : ', n_features)

                fs_class.n_features_to_select = n_features
                svc_train_data = fs_class.transform(train_data)
                svc_test_data = fs_class.transform(test_data)

                for r in range(reps):
                    model = train_Keras(svc_train_data, train_labels, svc_test_data, test_labels, model_kwargs)
                    train_data_norm = model.normalization.transform(svc_train_data)
                    test_data_norm = model.normalization.transform(svc_test_data)
                    test_pred = model.predict(test_data_norm)
                    n_BAs.append(balance_accuracy(test_labels, test_pred))
                    n_mAPs.append(average_precision_score(test_labels, test_pred))
                    n_accuracies.append(model.evaluate(test_data_norm, test_labels, verbose=0)[-1])
                    n_train_accuracies.append(model.evaluate(train_data_norm, train_labels, verbose=0)[-1])
                    del model
                    K.clear_session()
                    print(
                        'n_features : ', n_features,
                        ', acc : ', n_accuracies[-1],
                        ', BA : ', n_BAs[-1],
                        ', mAP : ', n_mAPs[-1],
                        ', train_acc : ', n_train_accuracies[-1],
                    )
                if i >= len(accuracies):
                    accuracies.append(n_accuracies)
                    BAs.append(n_BAs)
                    mAPs.append(n_mAPs)
                    nfeats.append(n_features)
                else:
                    accuracies[i] += n_accuracies
                    BAs[i] += n_BAs
                    mAPs[i] += n_mAPs

        mean_accuracies = np.array(accuracies).mean(axis=-1)
        print('NFEATS : ', nfeats)
        diff_accuracies = .5 * (mean_accuracies[1:] + mean_accuracies[:-1])
        np_nfeats = np.array(nfeats)
        diff = np_nfeats[1:] - np_nfeats[:-1]
        AUC = np.sum(diff * diff_accuracies) / np.sum(diff)
        print('AUC : ', AUC)

        output_filename = directory + 'three_layer_nn_' + fs_method.__name__ + '.json'

        if not os.path.isdir(directory):
            os.makedirs(directory)

        info_data = {
            'reps': reps,
            'classification': {
                'regularization': regularization,
                'n_features': nfeats,
                'accuracy': accuracies,
                'mean_accuracy': np.array(accuracies).mean(axis=1).tolist(),
                'BA': BAs,
                'mean_BA': np.array(BAs).mean(axis=1).tolist(),
                'mAP': mAPs,
                'mean_mAP': np.array(mAPs).mean(axis=1).tolist(),
                'auc': AUC
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
