from keras.utils import to_categorical
from keras import callbacks, regularizers, layers, models, optimizers
from keras.models import load_model
from keras.datasets import mnist
from src.wrn import network_models
import json
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from src.layers import e2efs
from keras import backend as K
import tensorflow as tf
import time


batch_size = 128
regularization = 5e-4
reps = 5
verbose = 2
warming_up = True

directory = os.path.dirname(os.path.realpath(__file__)) + '/info/'
fs_network = 'three_layer_nn'
classifier_network = 'wrn164'


def create_rank(scores, k):
    """
    Compute rank of each feature based on weight.

    """
    scores = abs(scores)
    n, d = scores.shape
    ranks = []
    for i, score in enumerate(scores):
        # Random permutation to avoid bias due to equal weights.
        idx = np.random.permutation(d)
        permutated_weights = score[idx]
        permutated_rank = (-permutated_weights).argsort().argsort() + 1
        rank = permutated_rank[np.argsort(idx)]

        ranks.append(rank)

    return np.array(ranks)

def compute_median_rank(scores, k):
    ranks = create_rank(scores, k)
    median_ranks = np.median(ranks[:,:k], axis = 1)
    return median_ranks


class Sample_Concrete(layers.Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables.

    """

    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):
        # logits: [BATCH_SIZE, d]
        logits_ = K.expand_dims(logits, -2)  # [BATCH_SIZE, 1, d]

        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        uniform = tf.random.uniform(shape=(batch_size, self.k, d),
                                    minval=np.finfo(tf.float32.as_numpy_dtype).tiny,
                                    maxval=1.0)

        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis=1)

        # Explanation Stage output.
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        return K.in_train_phase(samples, discrete_logits)

    def compute_output_shape(self, input_shape):
        return input_shape


def get_l2x_model(input_shape, nfeatures):
    activation = 'relu'
    # P(S|X)
    model_input = layers.Input(shape=input_shape, dtype='float32')

    net = layers.Flatten()(model_input)
    net = layers.Dense(100, activation=activation, name='s/dense1',
                kernel_regularizer=regularizers.l2(1e-3))(net)
    net = layers.Dense(100, activation=activation, name='s/dense2',
                kernel_regularizer=regularizers.l2(1e-3))(net)

    # A tensor of shape, [batch_size, max_sents, 100]
    logits = layers.Dense(np.prod(input_shape))(net)
    # [BATCH_SIZE, max_sents, 1]
    tau = 0.1
    samples = Sample_Concrete(tau, nfeatures, name='sample')(logits)
    samples = layers.Reshape(input_shape)(samples)
    return models.Model(model_input, samples)


def scheduler(extra=0, factor=1.):
    def sch(epoch):
        if epoch < 60 + extra:
            return .1 * factor
        elif epoch < 80 + extra:
            return .02 * factor
        elif epoch < 100 + extra:
            return .004 * factor
        else:
            return .0008 * factor
    return sch


def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    generator = ImageDataGenerator(
        # horizontal_flip=True,
        # height_shift_range=5./32.,
        # width_shift_range=5./32.,
        # fill_mode='reflect'
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
    name = 'mnist_' + fs_network + '_' + classifier_network + '_r_' + str(regularization)
    print(name)
    model_kwargs = {
        'nclasses': num_classes,
        'regularization': regularization
    }

    total_features = int(np.prod(train_data.shape[1:]))

    model_filename = directory + fs_network + '_trained_model.h5'
    classifier_filename = directory + classifier_network + '_trained_model.h5'
    if not os.path.exists(model_filename) and warming_up:

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
        if not os.path.isdir(directory):
            os.makedirs(directory)
        model.save(model_filename)
        del model
        K.clear_session()

    nfeats = []
    accuracies = []
    times = []

    for factor in [.05, .1, .25, .5]:
        n_features = int(total_features * factor)
        n_accuracies = []
        n_times = []

        for r in range(reps):
            print('factor : ', factor, ' , rep : ', r)
            l2x_model = get_l2x_model(train_data.shape[1:], n_features)
            classifier = load_model(model_filename) if warming_up else getattr(network_models, fs_network)(input_shape=train_data.shape[1:], **model_kwargs)
            classifier_input = layers.Multiply()([l2x_model.input, l2x_model.output])
            output = classifier(classifier_input)
            model = models.Model(l2x_model.input, output)

            optimizer = optimizers.adam(lr=1e-3)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
            model.classifier = classifier
            model.summary()
            start_time = time.time()
            model.fit_generator(
                generator.flow(train_data, train_labels, **generator_kwargs),
                steps_per_epoch=train_data.shape[0] // batch_size, epochs=80,
                callbacks=[],
                validation_data=(test_data, test_labels),
                validation_steps=test_data.shape[0] // batch_size,
                verbose=verbose
            )
            scores = l2x_model.predict(train_data, verbose=0, batch_size=batch_size).reshape((-1, np.prod(train_data.shape[1:]))).sum(axis=0)
            pos = np.argsort(scores)[::-1][:n_features]
            n_times.append(time.time() - start_time)
            mask = np.zeros_like(scores)
            mask[pos] = 1.
            mask = mask.reshape(train_data.shape[1:])
            del l2x_model, classifier, model
            K.clear_session()
            classifier = load_model(classifier_filename) if warming_up else getattr(network_models, classifier_network)(
                input_shape=train_data.shape[1:], **model_kwargs)
            optimizer = optimizers.SGD(lr=1e-1)
            classifier.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
            classifier.fit_generator(
                generator.flow(mask * train_data, train_labels, **generator_kwargs),
                steps_per_epoch=train_data.shape[0] // batch_size, epochs=80,
                callbacks=[
                    callbacks.LearningRateScheduler(scheduler(extra=0)),
                ],
                validation_data=(mask * test_data, test_labels),
                validation_steps=test_data.shape[0] // batch_size,
                verbose=verbose
            )

            n_accuracies.append(classifier.evaluate(mask * test_data, test_labels, verbose=0)[-1])
            del classifier
            K.clear_session()
        print(
            'n_features : ', n_features, ', acc : ', n_accuracies, ', time : ', n_times
        )
        accuracies.append(n_accuracies)
        nfeats.append(n_features)
        times.append(n_times)

    output_filename = directory + fs_network + '_' + classifier_network + '_l2x_results_warming_' + str(warming_up) + '.json'

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
