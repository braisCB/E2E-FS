from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks, initializers, optimizers
from src import optimizers as custom_optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from src.wrn import network_models
import json
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.callbacks import E2EFSCallback
from src.layers import e2efs
from tensorflow.keras import backend as K
import tensorflow as tf
import time


batch_size = 128
regularization = 5e-4
fs_reps = 1
reps = 5
verbose = 2
warming_up = True

directory = os.path.dirname(os.path.realpath(__file__)) + '/info/'
temp_directory = os.path.dirname(os.path.realpath(__file__)) + '/temp/'
network_names = ['densenet', ]
e2efs_classes = [e2efs.E2EFSRanking]


def scheduler(extra=0, factor=.01):
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
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    generator_fs = ImageDataGenerator(
        width_shift_range=5./32.,
        height_shift_range=5./32.,
        fill_mode='reflect',
        horizontal_flip=True,
    )
    generator = ImageDataGenerator(
        width_shift_range=5./32.,
        height_shift_range=5./32.,
        fill_mode='reflect',
        horizontal_flip=True
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


def main():

    dataset = load_dataset()

    for network_name in network_names:

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
        name = 'cifar10_' + network_name + '_r_' + str(regularization)
        print(name)
        model_kwargs = {
            'nclasses': num_classes,
            'regularization': regularization
        }

        total_features = int(np.prod(train_data.shape[1:]))

        model_filename = directory + network_name + '_trained_model.h5'
        if not os.path.isdir(directory):
            os.makedirs(directory)
        if not os.path.exists(model_filename) and warming_up:
            np.random.seed(1001)
            tf.set_random_seed(1001)
            model = getattr(network_models, network_name)(input_shape=train_data.shape[1:], **model_kwargs)
            print('training_model')
            model.fit_generator(
                generator.flow(train_data, train_labels, **generator_kwargs),
                steps_per_epoch=train_data.shape[0] // batch_size, epochs=80,
                callbacks=[
                    callbacks.LearningRateScheduler(scheduler(factor=1.))
                ],
                validation_data=(test_data, test_labels),
                validation_steps=test_data.shape[0] // batch_size,
                verbose=verbose
            )

            model.save(model_filename)
            del model
            K.clear_session()

        for e2efs_class in e2efs_classes:
            nfeats = []
            accuracies = []
            times = []

            cont_seed = 0
            for r in range(reps):
                temp_filename = temp_directory + network_name + '_' + e2efs_class.__name__ + \
                              '_e2efs_heatmap_iter_' + str(r) + '.npy'
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
                        classifier = load_model(model_filename) if warming_up else getattr(network_models, network_name)(
                            input_shape=train_data.shape[1:], **model_kwargs)
                        e2efs_layer = e2efs_class(1, input_shape=train_data.shape[1:],)
                                                  # kernel_initializer=initializers.constant(mask))
                        model = e2efs_layer.add_to_model(classifier, input_shape=train_data.shape[1:])

                        # optimizer = custom_optimizers.E2EFS_SGD(e2efs_layer=e2efs_layer, lr=1e-1)  # optimizers.adam(lr=1e-2)
                        optimizer = custom_optimizers.E2EFS_RMSprop(e2efs_layer=e2efs_layer, lr=1e-3)
                        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
                        model.fs_layer = e2efs_layer
                        model.classifier = classifier
                        model.summary()

                        model.fit_generator(
                            generator_fs.flow(train_data, train_labels, **generator_kwargs),
                            steps_per_epoch=train_data.shape[0] // batch_size, epochs=20000,
                            callbacks=[
                                E2EFSCallback(units=int(total_features * 0.05),
                                              verbose=verbose)
                            ],
                            validation_data=(test_data, test_labels),
                            validation_steps=test_data.shape[0] // batch_size,
                            verbose=verbose
                        )
                        heatmap += K.eval(model.heatmap)
                        del model
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
                    model = load_model(model_filename) if warming_up else getattr(network_models, network_name)(input_shape=train_data.shape[1:], **model_kwargs)
                    # optimizer = optimizers.SGD(lr=1e-1)  # optimizers.adam(lr=1e-2)
                    optimizer = optimizers.RMSprop(lr=1e-3)
                    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

                    model.fit_generator(
                        generator.flow(mask * train_data, train_labels, **generator_kwargs),
                        steps_per_epoch=train_data.shape[0] // batch_size, epochs=80,
                        callbacks=[
                            callbacks.LearningRateScheduler(scheduler()),
                        ],
                        validation_data=(mask * test_data, test_labels),
                        validation_steps=test_data.shape[0] // batch_size,
                        verbose=verbose
                    )
                    acc = model.evaluate(mask * test_data, test_labels, verbose=0)[-1]
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
