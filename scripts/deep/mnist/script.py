from keras.utils import to_categorical
from keras import callbacks, regularizers
from src import optimizers as custom_optimizers
from keras.models import load_model
from keras.datasets import mnist
from src.wrn import network_models
import json
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from src.callbacks import E2EFSCallback
from src.layers import e2efs
from keras import backend as K


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

batch_size = 128
regularization = 5e-4
reps = 3
verbose = 2
warming_up = True

directory = os.path.dirname(os.path.realpath(__file__)) + '/info/'
network_names = ['wrn164', ]
e2efs_classes = [
    (e2efs.E2EFSSoft, {'dropout': .1, 'decay_factor': .75, 'kernel_regularizer': regularizers.l2(1e-2)}),
    (e2efs.E2EFSHard, {'dropout': .1, 'l1': 1e0, 'l2': 1e0}),
]


def e2efs_factor(T=250):
    def func(epoch):
        if epoch < 5:
            return 0., 0., .5
        elif epoch < 140:
            return 1., (epoch - 5) / T, .5
        else:
            return 1., 1., 0.
    return func


def e2efs_units(units, input_shape, nepochs):
    p = (units / input_shape) ** (1. / nepochs)

    def func(epoch):
        return max(units, input_shape*p**epoch)

    return func


def scheduler(extra=0, factor=1.):
    def sch(epoch):
        if epoch < 30 + extra:
            return .1 * factor
        elif epoch < 60 + extra:
            return .02 * factor
        elif epoch < 90 + extra:
            return .004 * factor
        else:
            return .0008 * factor
    return sch


def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    generator = ImageDataGenerator(
        horizontal_flip=True,
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
        'generator': generator
    }
    return output


def main():

    dataset = load_dataset()

    for network_name in network_names:

        train_data = np.asarray(dataset['train']['data'])
        train_labels = dataset['train']['label']
        num_classes = len(np.unique(train_labels))

        test_data = np.asarray(dataset['test']['data'])
        test_labels = dataset['test']['label']

        train_labels = to_categorical(train_labels, num_classes=num_classes)
        test_labels = to_categorical(test_labels, num_classes=num_classes)

        generator = dataset['generator']
        generator_kwargs = {
            'batch_size': batch_size
        }

        print('reps : ', reps)
        name = 'mnist_' + network_name + '_r_' + str(regularization)
        print(name)
        model_kwargs = {
            'nclasses': num_classes,
            'regularization': regularization
        }

        total_features = int(np.prod(train_data.shape[1:]))

        model_filename = directory + network_name + '_trained_model.h5'
        if not os.path.exists(model_filename) and warming_up:

            model = getattr(network_models, network_name)(input_shape=train_data.shape[1:], **model_kwargs)
            print('training_model')
            model.fit_generator(
                generator.flow(train_data, train_labels, **generator_kwargs),
                steps_per_epoch=train_data.shape[0] // batch_size, epochs=110,
                callbacks=[
                    callbacks.LearningRateScheduler(scheduler())
                ],
                validation_data=(test_data, test_labels),
                validation_steps=test_data.shape[0] // batch_size,
                verbose=verbose
            )
            if not os.path.isdir(directory):
                os.makedirs(directory)
            model.save(model_filename)
            del model
            K.clear_session()

        for e2efs_class, e2efs_kwargs in e2efs_classes:
            nfeats = []
            accuracies = []

            for factor in [.05, .1, .25, .5]:
                n_features = int(total_features * factor)
                n_accuracies = []
                n_heatmaps = []

                for r in range(reps):
                    print('factor : ', factor, ' , rep : ', r)
                    classifier = load_model(model_filename) if warming_up else getattr(network_models, network_name)(input_shape=train_data.shape[1:], **model_kwargs)
                    e2efs_layer = e2efs_class(n_features, input_shape=train_data.shape[1:], **e2efs_kwargs)
                    model = e2efs_layer.add_to_model(classifier, input_shape=train_data.shape[1:])

                    optimizer = custom_optimizers.E2EFS_SGD(e2efs_layer=e2efs_layer, lr=1e-1)  # optimizers.adam(lr=1e-2)
                    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
                    model.fs_layer = e2efs_layer
                    model.classifier = classifier
                    model.summary()
                    model.fit_generator(
                        generator.flow(train_data, train_labels, **generator_kwargs),
                        steps_per_epoch=train_data.shape[0] // batch_size, epochs=190,
                        callbacks=[
                            callbacks.LearningRateScheduler(scheduler(extra=80)),
                            E2EFSCallback(factor_func=e2efs_factor(),
                                              units_func=e2efs_units(
                                                  n_features, total_features, 100
                                              ) if 'hard' in e2efs_class.__name__.lower() else None,
                                              verbose=verbose)
                        ],
                        validation_data=(test_data, test_labels),
                        validation_steps=test_data.shape[0] // batch_size,
                        verbose=verbose
                    )
                    n_heatmaps.append(K.eval(model.heatmap).tolist())
                    n_accuracies.append(model.evaluate(test_data, test_labels, verbose=0)[-1])
                    del model
                    K.clear_session()
                print(
                    'n_features : ', n_features, ', acc : ', n_accuracies
                )
                accuracies.append(n_accuracies)
                nfeats.append(n_features)

            output_filename = directory + network_name + '_' + e2efs_class.__name__ + \
                              '_e2efs_results_warming_' + str(warming_up) + '.json'

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
                        'accuracy': accuracies
                    }
                }
            )

            with open(output_filename, 'w') as outfile:
                json.dump(info_data, outfile)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../../../')
    main()