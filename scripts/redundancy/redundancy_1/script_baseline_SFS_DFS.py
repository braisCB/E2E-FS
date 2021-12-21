from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks, optimizers as keras_optimizers
import json
import numpy as np
import os
from dataset_reader import redundancy_1
from src.network_models import three_layer_nn
from src.baseline_methods import SFS, DFS
from sklearn.model_selection import RepeatedStratifiedKFold
import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


fs_methods = [
    DFS.DFS,
    SFS.SFS,
    SFS.iSFS
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
normalization_func = redundancy_1.Normalize

dataset_name = 'redundancy_1'
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
    dataset = redundancy_1.load_dataset()
    return dataset


def create_model(input_size, num_classes=2, dfs=False, **kwargs):
    model = three_layer_nn(nfeatures=input_size, nclasses=num_classes, dfs=dfs, **kwargs)

    optimizer = optimizer_class(lr=initial_lr)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['acc']
    )
    return model


def get_fit_kwargs(train_y):
    num_samples, num_classes = train_y.shape
    class_weight = train_y.shape[0] / np.sum(train_y, axis=0)
    class_weight = num_classes * class_weight / class_weight.sum()
    sample_weight = np.zeros((num_samples,))
    sample_weight[train_y[:, 1] == 1] = class_weight[1]
    sample_weight[train_y[:, 1] == 0] = class_weight[0]
    batch_size = max(2, len(train_y) // 50)
    return dict(
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            callbacks.LearningRateScheduler(scheduler()),
        ],
        sample_weight=sample_weight,
        verbose=verbose
    )



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
        fs_time = []
        mus = []
        real_feats = []
        redundant_feats = []
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
            }
            print('batch_size :', batch_size)

            keras_model = lambda x: create_model(x, num_classes=num_classes,
                                                 dfs='dfs' in fs_method.__name__.lower(), **model_kwargs)

            print('Starting feature selection')
            fs_dir = os.path.dirname(os.path.realpath(__file__)) + '/temp/'
            if not os.path.isdir(fs_dir):
                os.makedirs(fs_dir)
            fs_filename = fs_dir + fs_method.__name__ + '_iter_' + str(j) + '_seed_' + \
                          str(random_state) + '.json'
            if os.path.exists(fs_filename):
                with open(fs_filename, 'r') as outfile:
                    fs_data = json.load(outfile)
                fs_class = fs_method(model_func=keras_model, n_features_to_select=10)
                fs_class.score = np.asarray(fs_data['score'])
                fs_class.ranking = np.asarray(fs_data['ranking'])
                fs_time.append(np.NAN)
            else:
                norm = normalization_func()
                train_data_norm = norm.fit_transform(train_data)
                test_data_norm = norm.transform(test_data)

                start_time = time.process_time()
                fs_class = fs_method(model_func=keras_model, n_features_to_select=10)
                fit_kwargs = get_fit_kwargs(train_labels)
                fs_class.fit(train_data_norm, train_labels, **fit_kwargs)
                fs_data = {
                    'score': fs_class.score.tolist(),
                    'ranking': fs_class.ranking.tolist()
                }
                fs_time.append(time.process_time() - start_time)
                with open(fs_filename, 'w') as outfile:
                    json.dump(fs_data, outfile)
            print('Finishing feature selection. Time : ', fs_time[-1], 's')


            for i, n_features in enumerate([20, 40, 60, 80, 100]):
                rf, rdf = redundancy_1.get_redundancy_stats(fs_class.ranking[:n_features])
                n_real_feats = [rf]
                n_redundant_feats = [rdf]
                print('n_features : ', n_features)
                print('real_feats : ', n_real_feats[-1])
                print('redundant_feats : ', n_redundant_feats[-1])

                if i >= len(real_feats):
                    real_feats.append(n_real_feats)
                    redundant_feats.append(n_redundant_feats)
                else:
                    real_feats[i] += n_real_feats
                    redundant_feats[i] += n_redundant_feats

        output_filename = directory + 'three_layer_nn_' + fs_method.__name__ + '.json'

        if not os.path.isdir(directory):
            os.makedirs(directory)

        info_data = {
            'reps': reps,
            'classification': {
                'mus': mus,
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