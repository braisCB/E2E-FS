from keras.utils import to_categorical
from keras import callbacks, initializers, optimizers
from keras.models import load_model
from keras.datasets import fashion_mnist
from src.wrn import network_models
import json
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from e2efs.callbacks import E2EFSCallback
from keras import backend as K
import tensorflow as tf
import time
if tf.__version__ < '2.0':
    from e2efs import optimizers as custom_optimizers
    from e2efs import e2efs_layers
else:
    from e2efs import optimizers_tf2 as custom_optimizers
    from e2efs import e2efs_layers_tf2 as e2efs_layers
    tf.set_random_seed = tf.random.set_seed


batch_size = 128
regularization = 5e-4
reps = 5
verbose = 2
warming_up = True

directory = os.path.dirname(os.path.realpath(__file__)) + '/info/'
classifier_network = 'efficientnetB0'
fs_network = 'three_layer_nn'
e2efs_classes = [e2efs_layers.E2EFS, e2efs_layers.E2EFSSoft]


def scheduler(extra=0, factor=.1):
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
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    fs_generator = ImageDataGenerator(
        # width_shift_range=5./32.,
        # height_shift_range=5./32.,
        # fill_mode='reflect',
        horizontal_flip=True,
    )
    generator = ImageDataGenerator(
        # width_shift_range=5. / 32.,
        # height_shift_range=5. / 32.,
        # fill_mode='reflect',
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
        'generator': generator,
        'fs_generator': fs_generator
    }
    return output


def main():

    dataset = load_dataset()

    train_data = np.asarray(dataset['train']['data'])
    train_labels = dataset['train']['label']
    num_classes = len(np.unique(train_labels))

    test_data = np.asarray(dataset['test']['data'])
    test_labels = dataset['test']['label']

    train_labels = to_categorical(train_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)

    generator = dataset['generator']
    fs_generator = dataset['fs_generator']
    generator_kwargs = {
        'batch_size': batch_size
    }

    print('reps : ', reps)
    name = 'fashion_mnist_' + fs_network + '_r_' + str(regularization)
    print(name)
    model_kwargs = {
        'nclasses': num_classes,
        'regularization': regularization
    }

    total_features = int(np.prod(train_data.shape[1:]))

    fs_filename = directory + fs_network + '_trained_model.h5'
    classifier_filename = directory + classifier_network + '_trained_model.h5'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if not os.path.exists(fs_filename) and warming_up:
        np.random.seed(1001)
        tf.set_random_seed(1001)
        model = getattr(network_models, fs_network)(input_shape=train_data.shape[1:], **model_kwargs)
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

        model.save(fs_filename)
        del model
        K.clear_session()

    for e2efs_class in e2efs_classes:
        nfeats = []
        accuracies = []
        times = []

        cont_seed = 0

        for factor in [.05, .1, .25, .5]:
            n_features = int(total_features * factor)
            n_accuracies = []
            n_times = []

            for r in range(reps):
                print('factor : ', factor, ' , rep : ', r)
                np.random.seed(cont_seed)
                tf.set_random_seed(cont_seed)
                cont_seed += 1
                mask = (np.std(train_data, axis=0) > 1e-3).astype(int).flatten()
                classifier = load_model(fs_filename) if warming_up else getattr(network_models, fs_network)(input_shape=train_data.shape[1:], **model_kwargs)
                e2efs_layer = e2efs_class(n_features, input_shape=train_data.shape[1:], kernel_initializer=initializers.constant(mask))
                model = e2efs_layer.add_to_model(classifier, input_shape=train_data.shape[1:])

                # optimizer = custom_optimizers.E2EFS_SGD(e2efs_layer=e2efs_layer, lr=1e-1)  # optimizers.adam(lr=1e-2)
                optimizer = custom_optimizers.E2EFS_Adam(e2efs_layer=e2efs_layer, lr=1e-2)
                model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
                model.fs_layer = e2efs_layer
                model.classifier = classifier
                model.summary()
                start_time = time.time()
                model.fit_generator(
                    fs_generator.flow(train_data, train_labels, **generator_kwargs),
                    steps_per_epoch=train_data.shape[0] // batch_size, epochs=20000,
                    callbacks=[
                        E2EFSCallback(verbose=verbose)
                    ],
                    validation_data=(test_data, test_labels),
                    validation_steps=test_data.shape[0] // batch_size,
                    verbose=verbose
                )
                fs_rank = np.argsort(K.eval(model.heatmap))[::-1]
                mask = np.zeros(train_data.shape[1:])
                mask.flat[fs_rank[:n_features]] = 1.
                # mask = K.eval(model.fs_kernel).reshape(train_data.shape[1:])
                n_times.append(time.time() - start_time)
                print('nnz : ', np.count_nonzero(mask))
                del model
                K.clear_session()
                model = load_model(classifier_filename) if warming_up else getattr(network_models, classifier_network)(
                    input_shape=train_data.shape[1:], **model_kwargs)
                # optimizer = optimizers.RMSprop(learning_rate=1e-2) # optimizers.SGD(lr=1e-1)  # optimizers.adam(lr=1e-2)
                optimizer = optimizers.Adam(learning_rate=1e-2)
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
                n_accuracies.append(model.evaluate(mask * test_data, test_labels, verbose=0)[-1])
                del model
                K.clear_session()
            print(
                'n_features : ', n_features, ', acc : ', n_accuracies, ', time : ', n_times
            )
            accuracies.append(n_accuracies)
            nfeats.append(n_features)
            times.append(n_times)

        output_filename = directory + fs_network + '_' + classifier_network + '_' + e2efs_class.__name__ + \
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
