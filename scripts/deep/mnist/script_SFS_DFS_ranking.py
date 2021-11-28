from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks, initializers, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from src.wrn import network_models
import json
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from src.baseline_methods import SFS, DFS
import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
if tf.__version__ < '2.0':
    from src import optimizers as custom_optimizers
else:
    tf.set_random_seed = tf.random.set_seed


batch_size = 128
regularization = 5e-4
fs_reps = 1
reps = 5
verbose = 2
warming_up = True

directory = os.path.dirname(os.path.realpath(__file__)) + '/info/'
temp_directory = os.path.dirname(os.path.realpath(__file__)) + '/temp/'
fs_network = 'efficientnetB0'
fs_classes = [DFS.DFS, SFS.SFS]


def scheduler(extra=0, factor=1.):
    def sch(epoch):
        if epoch < 30 + extra:
            return .1 * factor
        elif epoch < 50 + extra:
            return .02 * factor
        elif epoch < 70 + extra:
            return .004 * factor
        else:
            return .0008 * factor
    return sch


def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    generator_fs = ImageDataGenerator(
        # width_shift_range=4./28.,
        # height_shift_range=4./28.,
        # fill_mode='reflect',
        # horizontal_flip=True,
    )
    generator = ImageDataGenerator(
        # width_shift_range=4./28.,
        # height_shift_range=4./28.,
        # fill_mode='reflect',
        # horizontal_flip=True
    )
    y_train = np.reshape(y_train, [-1, 1])
    y_test = np.reshape(y_test, [-1, 1])
    x_train = x_train / 127.5 - 1.
    x_test = x_test / 127.5 - 1.
    output = {
        'train': {
            'data': x_train,
            'label': y_train
        },
        'test': {
            'data': x_test,
            'label': y_test
        },
        'generator': generator,
        'generator_fs': generator_fs
    }
    return output


def get_fit_kwargs(train_label):
    return dict(
        steps_per_epoch=train_label.shape[0] // batch_size, epochs=80,
        callbacks=[
            callbacks.LearningRateScheduler(scheduler()),
        ],
        verbose=verbose
    )


def main():

    dataset = load_dataset()

    train_data = np.asarray(dataset['train']['data'])
    train_labels = dataset['train']['label']
    num_classes = len(np.unique(train_labels))

    # mask = (np.std(train_data, axis=0) > 5e-3).astype(int).flatten()

    test_data = np.asarray(dataset['test']['data'])
    test_labels = dataset['test']['label']

    train_labels = to_categorical(train_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)

    generator = dataset['generator']
    generator_fs = dataset['generator_fs']
    generator_kwargs = {
        'batch_size': batch_size
    }

    print('reps : ', reps)
    name = 'mnist_' + fs_network + '_r_' + str(regularization)
    print(name)
    model_kwargs = {
        'nclasses': num_classes,
        'regularization': regularization
    }

    total_features = int(np.prod(train_data.shape[1:]))

    filename = directory + fs_network + '_trained_model.h5'

    if not os.path.isdir(directory):
        os.makedirs(directory)
    if not os.path.exists(filename) and warming_up:
        np.random.seed(1001)
        tf.set_random_seed(1001)
        model = getattr(network_models, fs_network)(input_shape=train_data.shape[1:], **model_kwargs)
        print('training_model')
        model.fit_generator(
            generator.flow(train_data, train_labels, **generator_kwargs),
            steps_per_epoch=train_data.shape[0] // batch_size, epochs=80,
            callbacks=[
                callbacks.LearningRateScheduler(scheduler())
            ],
            validation_data=(test_data, test_labels),
            validation_steps=test_data.shape[0] // batch_size,
            verbose=verbose
        )

        model.save(filename)
        del model
        K.clear_session()

    for fs_class in fs_classes:
        nfeats = []
        accuracies = []
        times = []

        cont_seed = 0
        for r in range(reps):
            temp_filename = temp_directory + fs_network + '_' + fs_class.__name__ + \
                          '_heatmap_iter_' + str(r) + '.npy'
            if os.path.exists(temp_filename):
                heatmap = np.load(temp_filename)
            else:
                heatmap = np.zeros(np.prod(train_data.shape[1:]))
                start_time = time.time()
                for fs_r in range(fs_reps):
                    print('rep : ', fs_r)
                    np.random.seed(cont_seed)
                    tf.set_random_seed(cont_seed)
                    cont_seed += 1

                    classifier = lambda x: getattr(network_models, fs_network)(
                        input_shape=x, dfs='dfs' in fs_class.__name__.lower(), **model_kwargs)

                    fs_method = fs_class(model_func=classifier, n_features_to_select=int(total_features * .05))
                    fit_kwargs = get_fit_kwargs(train_labels)
                    fs_method.fit(train_data, train_labels, {'generator_kwargs': generator_kwargs, 'generator': generator_fs}, **fit_kwargs)

                    heatmap += fs_method.score
                    del fs_method
                    K.clear_session()
                if not os.path.isdir(temp_directory):
                    os.makedirs(temp_directory)
                np.save(temp_filename, heatmap)
                times.append(time.time() - start_time)
            fs_rank = np.argsort(heatmap)[::-1]

            for i, factor in enumerate([.05, .1, .25, .5]):
                print('factor : ', factor, ' , rep : ', r)
                n_features = int(total_features * factor)
                mask = np.zeros(train_data.shape[1:])
                mask.flat[fs_rank[:n_features]] = 1.

                np.random.seed(cont_seed)
                tf.set_random_seed(cont_seed)
                cont_seed += 1
                model = load_model(filename) if warming_up else getattr(network_models, fs_network)(input_shape=train_data.shape[1:], **model_kwargs)
                # optimizer = optimizers.RMSprop(learning_rate=1e-2)  # optimizers.SGD(lr=1e-1)  # optimizers.adam(lr=1e-2)
                optimizer = optimizers.SGD(learning_rate=1e-2)
                model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

                model.fit_generator(
                    generator.flow(mask * train_data, train_labels, **generator_kwargs),
                    steps_per_epoch=train_data.shape[0] // batch_size, epochs=80,
                    callbacks=[
                        callbacks.LearningRateScheduler(scheduler()),
                    ],
                    verbose=verbose
                )
                acc = float(model.evaluate(mask * test_data, test_labels, verbose=0)[-1])
                if i < len(accuracies):
                    accuracies[i].append(acc)
                else:
                    accuracies.append([acc])
                    nfeats.append(n_features)
                del model
                K.clear_session()
                print(
                    'n_features : ', n_features, ', acc : ', acc, ', time : ', times[-1]
                )

        output_filename = directory + fs_network + fs_class.__name__ + \
                          '_results_warming_' + str(warming_up) + '.json'

        try:
            with open(output_filename) as outfile:
                info_data = json.load(outfile)
        except:
            info_data = {}

        if name not in info_data:
            info_data[name] = []

        info_data[name].append(
            {
                'regularization': regularization,
                'reps': reps,
                'classification': {
                    'n_features': nfeats,
                    'accuracy': accuracies,
                    'times': times
                }
            }
        )

        with open(output_filename, 'w') as outfile:
            json.dump(info_data, outfile)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../../../')
    main()
