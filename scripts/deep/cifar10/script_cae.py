from keras.utils import to_categorical
from keras import callbacks, initializers, layers, models, optimizers
from keras.models import load_model
from keras.datasets import cifar10
from src.wrn import network_models
import json
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import time


batch_size = 128
regularization = 5e-4
reps = 5
verbose = 2
warming_up = True

directory = os.path.dirname(os.path.realpath(__file__)) + '/info/'
network_names = ['wrn164', ]


class ConcreteSelect(layers.Layer):

    def __init__(self, output_dim, start_temp=10.0, min_temp=0.1, alpha=0.99999, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        super(ConcreteSelect, self).__init__(**kwargs)

    def build(self, input_shape):
        self.temp = self.add_weight(name='temp', shape=[], initializer=initializers.Constant(self.start_temp), trainable=False)
        self.logits = self.add_weight(name='logits', shape=[self.output_dim, input_shape[1]],
                                      initializer=initializers.glorot_normal(), trainable=True)
        super(ConcreteSelect, self).build(input_shape)

    def call(self, X, training=None):
        uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0)
        gumbel = -K.log(-K.log(uniform))
        temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
        noisy_logits = (self.logits + gumbel) / temp
        samples = K.softmax(noisy_logits)

        discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1])

        self.selections = K.in_train_phase(samples, discrete_logits, training)
        Y = K.dot(X, K.transpose(self.selections))
        # Y = K.sum(self.selections, axis=0) * X
        return Y

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim
        # return input_shape


class StopperCallback(callbacks.EarlyStopping):

    def __init__(self, mean_max_target=0.998):
        self.mean_max_target = mean_max_target
        super(StopperCallback, self).__init__(monitor='', patience=float('inf'), verbose=1, mode='max',
                                              baseline=self.mean_max_target)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('mean max of probabilities:', self.get_monitor_value(logs), '- temperature',
                  K.get_value(self.model.get_layer('concrete_select').temp))
        # print( K.get_value(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
        # print(K.get_value(K.max(self.model.get_layer('concrete_select').selections, axis = -1)))

    def get_monitor_value(self, logs):
        monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis=-1)))
        return monitor_value


class ConcreteAutoencoderFeatureSelector():

    def __init__(self, K, output_function):
        self.K = K
        self.output_function = output_function

    def fit(self, generator, X, Y=None, val_X=None, val_Y=None, num_epochs=100, batch_size=None, start_temp=10.0,
            min_temp=0.1, tryout_limit=1, class_weight=None):
        if Y is None:
            Y = X
        assert len(X) == len(Y)
        validation_data = None
        if val_X is not None and val_Y is not None:
            assert len(val_X) == len(val_Y)
            validation_data = (val_X, val_Y)

        if batch_size is None:
            batch_size = max(len(X) // 256, 16)

        steps_per_epoch = (len(X) + batch_size - 1) // batch_size

        for i in range(tryout_limit):

            K.set_learning_phase(1)

            inputs = layers.Input(shape=X.shape[1:])
            x = layers.Flatten()(inputs)

            alpha = np.exp(np.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))

            self.concrete_select = ConcreteSelect(self.K, start_temp, min_temp, alpha, name='concrete_select')

            selected_features = self.concrete_select(x)

            outputs = self.output_function(selected_features)

            self.model = models.Model(inputs, outputs)

            self.model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizers.Adam(1e-1),
                metrics=['acc']
            )

            print(self.model.summary())

            stopper_callback = StopperCallback()

            hist = self.model.fit_generator(
                generator.flow(X, Y, batch_size=batch_size),
                steps_per_epoch=X.shape[0] // batch_size, epochs=80,
                callbacks=[StopperCallback(), callbacks.LearningRateScheduler(scheduler(80, .1))],
                validation_data=validation_data,
                verbose=2
                # validation_steps=test_data.shape[0] // batch_size,
            )  # , validation_freq = 10)

            if K.get_value(
                    K.mean(K.max(K.softmax(self.concrete_select.logits, axis=-1)))) >= stopper_callback.mean_max_target:
                break

            num_epochs *= 2

        self.probabilities = K.get_value(K.softmax(self.model.get_layer('concrete_select').logits))
        self.indices = K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))

        return self

    def get_indices(self):
        return K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))

    def get_mask(self):
        return K.get_value(K.sum(K.one_hot(K.argmax(self.model.get_layer('concrete_select').logits),
                                           self.model.get_layer('concrete_select').logits.shape[1]), axis=0))

    def transform(self, X):
        return X[self.get_indices()]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices=False):
        return self.get_indices() if indices else self.get_mask()

    def get_params(self):
        return self.model


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
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    generator = ImageDataGenerator(
        horizontal_flip=True,
        height_shift_range=5./32.,
        width_shift_range=5./32.,
        fill_mode='reflect'
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
        name = 'cifar10_' + network_name + '_r_' + str(regularization)
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

        nfeats = []
        accuracies = []
        model_accuracies = []
        times = []

        for factor in [.05, .1, .25, .5]:
            n_features = int(total_features * factor)
            n_accuracies = []
            n_model_accuracies = []
            n_times = []

            for r in range(reps):
                print('factor : ', factor, ' , rep : ', r)
                classifier = load_model(model_filename) if warming_up else getattr(network_models, network_name)(input_shape=train_data.shape[1:], **model_kwargs)
                cae_model = ConcreteAutoencoderFeatureSelector(K=n_features, output_function=classifier)

                start_time = time.time()
                cae_model.fit(generator, train_data, train_labels, test_data, test_labels, batch_size=batch_size)
                fs_rank = cae_model.indices
                n_times.append(time.time() - start_time)

                n_model_accuracies.append(cae_model.model.evaluate(test_data, test_labels, verbose=0)[-1])
                mask = np.zeros(train_data.shape[1:])
                mask.flat[fs_rank[:n_features]] = 1.
                del classifier, cae_model.model
                K.clear_session()
                classifier = load_model(model_filename) if warming_up else getattr(network_models, network_name)(
                    input_shape=train_data.shape[1:], **model_kwargs)
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
            model_accuracies.append(n_model_accuracies)
            nfeats.append(n_features)
            times.append(n_times)

        output_filename = directory + network_name + '_cae_results_warming_' + str(warming_up) + '.json'

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
                    'model_accuracy': model_accuracies,
                    'times': times
                }
            }
        )

        with open(output_filename, 'w') as outfile:
            json.dump(info_data, outfile)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../../../')
    main()
