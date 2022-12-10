from keras.utils import to_categorical
from keras import callbacks
import json
import numpy as np
import os
from dataset_reader import redundancy_1
from e2efs import e2efs_layers_tf2 as e2efs_layers
from src.network_models import three_layer_nn
from sklearn.model_selection import RepeatedStratifiedKFold
from keras import backend as K
from e2efs import callbacks as clbks, optimizers_tf2 as optimizers
import time
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


epochs = 150
reps = 1
verbose = 0
k_folds = 3
k_fold_reps = 20
optimizer_class = optimizers.E2EFS_Adam
normalization_func = redundancy_1.Normalize
regularization = 1e-3

dataset_name = 'redundancy_1'
directory = os.path.dirname(os.path.realpath(__file__)) + '/info/'
e2efs_classes = [e2efs_layers.E2EFS, e2efs_layers.E2EFSSoft]

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
    dataset = redundancy_1.load_dataset()
    return dataset


def train_Keras(train_X, train_y, test_X, test_y, kwargs, e2efs_class=None, n_features=None, epochs=150):
    normalization = normalization_func()
    num_classes = train_y.shape[-1]

    norm_train_X = normalization.fit_transform(train_X)
    norm_test_X = normalization.transform(test_X)

    batch_size = max(2, len(train_X) // 50)
    class_weight = train_y.shape[0] / np.sum(train_y, axis=0)
    class_weight = num_classes * class_weight / class_weight.sum()
    print('l2 :', kwargs['regularization'], ', batch_size :', batch_size)
    print('reps : ', reps, ', weights : ', class_weight)
    sample_weight = class_weight[np.argmax(train_y, axis=1)]
    class_weight = None

    classifier = three_layer_nn(nfeatures=norm_train_X.shape[1:], nclasses=num_classes, **kwargs)

    model_clbks = [
        callbacks.LearningRateScheduler(scheduler()),
    ]

    fs_callbacks = []

    if e2efs_class is not None:
        e2efs_layer = e2efs_class(n_features, input_shape=norm_train_X.shape[1:])
        model = e2efs_layer.add_to_model(classifier, input_shape=norm_train_X.shape[1:])
        fs_callbacks.append(
            clbks.E2EFSCallback(verbose=verbose)
        )
    else:
        model = classifier
        e2efs_layer = None

    optimizer = optimizer_class(e2efs_layer, lr=initial_lr)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['acc']
    )

    if e2efs_class is not None:
        model.fs_layer = e2efs_layer
        model.heatmap = e2efs_layer.moving_heatmap

        start_time = time.process_time()
        model.fit(
            norm_train_X, train_y, batch_size=batch_size,
            epochs=200000,
            callbacks=fs_callbacks,
            validation_data=(norm_test_X, test_y),
            class_weight=class_weight,
            sample_weight=sample_weight,
            verbose=verbose
        )
        model.fs_time = time.process_time() - start_time


    model.normalization = normalization

    return model


def main(dataset_name):

    dataset = load_dataset()

    raw_data = np.asarray(dataset['raw']['data'])
    raw_label = np.asarray(dataset['raw']['label'])
    num_classes = len(np.unique(raw_label))

    rskf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=k_fold_reps, random_state=42)

    for e2efs_class in e2efs_classes:
        print('E2EFS-Method : ', e2efs_class.__name__)
        cont_seed = 0

        nfeats = []
        fs_time = []
        real_feats = []
        redundant_feats = []
        name = dataset_name + '_three_layer_nn'
        print(name)

        for j, (train_index, test_index) in enumerate(rskf.split(raw_data, raw_label)):
            print('k_fold', j, 'of', k_folds*k_fold_reps)

            train_data, train_labels = raw_data[train_index], raw_label[train_index]
            test_data, test_labels = raw_data[test_index], raw_label[test_index]

            train_labels = to_categorical(train_labels, num_classes=num_classes)
            test_labels = to_categorical(test_labels, num_classes=num_classes)

            valid_features = np.where(np.abs(train_data).sum(axis=0) > 0)[0]
            if len(valid_features) < train_data.shape[1]:
                print('Removing', train_data.shape[1] - len(valid_features), 'zero features')
                train_data = train_data[:, valid_features]
                test_data = test_data[:, valid_features]

            model_kwargs = {
                'regularization': regularization,  # 100. / len(train_data), # regularization
            }

            for i, n_features in enumerate([20, 40, 60, 80, 100]):
                n_time = []
                n_real_feats = []
                n_redundant_feats = []
                print('n_features : ', n_features)

                heatmaps = []
                for r in range(reps):
                    np.random.seed(cont_seed)
                    tf.random.set_seed(cont_seed)
                    cont_seed += 1

                    model = train_Keras(
                        train_data, train_labels, test_data, test_labels, model_kwargs,
                        e2efs_class=e2efs_class, n_features=n_features,
                    )
                    heatmaps.append(K.eval(model.heatmap))
                    n_time.append(model.fs_time)
                    del model
                    K.clear_session()

                heatmap = np.mean(heatmaps, axis=0)
                best_features = np.argsort(heatmap)[::-1][:n_features]

                rf, rdf = redundancy_1.get_redundancy_stats(best_features)
                n_real_feats.append(rf)
                n_redundant_feats.append(rdf)
                print('real_feats : ', n_real_feats[-1])
                print('redundant_feats : ', n_redundant_feats[-1])

                if i >= len(real_feats):
                    fs_time.append(n_time)
                    nfeats.append(n_features)
                    real_feats.append(n_real_feats)
                    redundant_feats.append(n_redundant_feats)
                else:
                    fs_time[i] += n_time
                    real_feats[i] += n_real_feats
                    redundant_feats[i] += n_redundant_feats

        output_filename = directory + 'three_layer_nn_' + e2efs_class.__name__ + '.json'

        if not os.path.isdir(directory):
            os.makedirs(directory)

        info_data = {
            'reps': reps,
            'classification': {
                'regularization': regularization,
                'n_features': nfeats,
                'fs_time': fs_time,
                'real_feats': real_feats,
                'mean_real_feats': np.mean(real_feats, axis=1).tolist(),
                'redundant_feats': redundant_feats,
                'mean_redundant_feats': np.mean(redundant_feats, axis=1).tolist(),
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
