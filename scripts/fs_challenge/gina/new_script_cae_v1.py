from keras.utils import to_categorical
from keras import callbacks, initializers, layers, models, optimizers
import json
import numpy as np
import os
from dataset_reader import gina
from src.utils import balance_accuracy
from src.svc.models import LinearSVC
from extern.liblinear.python import liblinearutil
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score
from keras import backend as K
from src import callbacks as clbks
import time
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


epochs = 150
extra_epochs = 200
mu = 100
kernel = 'linear'
reps = 1
verbose = 0
loss_function = 'square_hinge'
k_folds = 3
k_fold_reps = 20
optimizer_class = optimizers.Adam
normalization_func = gina.Normalize

dataset_name = 'gina'
directory = os.path.dirname(os.path.realpath(__file__)) + '/info/'

initial_lr = .01


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

    def fit(self, X, Y=None, val_X=None, val_Y=None, num_epochs=300, batch_size=None, start_temp=10.0,
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

            alpha = np.exp(np.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))

            self.concrete_select = ConcreteSelect(self.K, start_temp, min_temp, alpha, name='concrete_select')

            selected_features = self.concrete_select(inputs)

            outputs = self.output_function(selected_features)

            self.model = models.Model(inputs, outputs)

            self.model.compile(
                loss=LinearSVC.loss_function(loss_function, class_weight),
                optimizer=optimizer_class(lr=initial_lr),
                metrics=[LinearSVC.accuracy]
            )

            print(self.model.summary())

            stopper_callback = StopperCallback()

            hist = self.model.fit(X, Y, batch_size, num_epochs, verbose=0, callbacks=[stopper_callback],
                                  validation_data=validation_data)  # , validation_freq = 10)

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


def scheduler(extra_epochs=0):
    def sch(epoch):
        if epoch < 50 + extra_epochs:
            return initial_lr
        elif epoch < 100 + extra_epochs:
            return .2 * initial_lr
        else:
            return .04 * initial_lr

    return sch


def load_dataset():
    dataset = gina.load_dataset()
    return dataset


def train_Keras(train_X, train_y, test_X, test_y, kwargs, cae_model_func=None, n_features=None, epochs=150):
    normalization = normalization_func()
    num_classes = train_y.shape[-1]

    norm_train_X = normalization.fit_transform(train_X)
    norm_test_X = normalization.transform(test_X)

    batch_size = max(2, len(train_X) // 50)
    class_weight = train_y.shape[0] / np.sum(train_y, axis=0)
    class_weight = num_classes * class_weight / class_weight.sum()
    sample_weight = None
    print('mu :', kwargs['mu'], ', batch_size :', batch_size)
    print('reps : ', reps, ', weights : ', class_weight)
    if num_classes == 2:
        sample_weight = np.zeros((len(norm_train_X),))
        sample_weight[train_y[:, 1] == 1] = class_weight[1]
        sample_weight[train_y[:, 1] == 0] = class_weight[0]
        class_weight = None

    model_clbks = [
        callbacks.LearningRateScheduler(scheduler()),
    ]


    if cae_model_func is not None:
        svc_model = LinearSVC(nfeatures=(n_features,), **kwargs)
        svc_model.create_keras_model(nclasses=num_classes)
        classifier = svc_model.model
        cae_model = cae_model_func(output_function=classifier, K=n_features)
        start_time = time.process_time()
        cae_model.fit(
            norm_train_X, train_y, norm_test_X, test_y, num_epochs=800, batch_size=batch_size,
            class_weight=class_weight
        )
        model = cae_model.model
        model.indices = cae_model.get_support(True)
        model.heatmap = cae_model.probabilities.max(axis=0)
        model.fs_time = time.process_time() - start_time
    else:
        svc_model = LinearSVC(norm_train_X.shape[1:], **kwargs)
        svc_model.create_keras_model(nclasses=num_classes)
        model = svc_model.model
        model.compile(
            loss=LinearSVC.loss_function(loss_function, class_weight),
            optimizer=optimizer_class(lr=initial_lr),
            metrics=[LinearSVC.accuracy]
        )
        model.fit(
            norm_train_X, train_y, batch_size=batch_size,
            epochs=epochs,
            callbacks=model_clbks,
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

    rskf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=k_fold_reps, random_state=42)

    print('L2X-Method')
    cont_seed = 0

    nfeats = []
    accuracies = []
    model_accuracies = []
    svc_accuracies = []
    fs_time = []
    BAs = []
    svc_BAs = []
    model_BAs = []
    mAPs = []
    svc_mAPs = []
    model_mAPs = []
    mus = []
    name = dataset_name + '_' + kernel + '_mu_' + str(mu)
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
            'mu': mu / len(train_data),
            'kernel': kernel,
            'degree': 3
        }


        svc_kwargs = {
            'C': 1.0,
            'solver': 0.
        }

        for i, n_features in enumerate([10, 50, 100, 150, 200]):
            n_accuracies = []
            n_svc_accuracies = []
            n_model_accuracies = []
            n_BAs = []
            n_svc_BAs = []
            n_model_BAs = []
            n_mAPs = []
            n_svc_mAPs = []
            n_model_mAPs = []
            n_train_accuracies = []
            n_time = []
            print('n_features : ', n_features)

            heatmaps = []
            for r in range(reps):
                np.random.seed(cont_seed)
                K.tf.set_random_seed(cont_seed)
                cont_seed += 1

                model = train_Keras(
                    train_data, train_labels, test_data, test_labels, model_kwargs,
                    cae_model_func=ConcreteAutoencoderFeatureSelector, n_features=n_features,
                )
                heatmaps.append(model.heatmap)
                n_time.append(model.fs_time)
                test_data_norm = model.normalization.transform(test_data)
                train_data_norm = model.normalization.transform(train_data)
                test_pred = model.predict(test_data_norm)
                n_model_accuracies.append(model.evaluate(test_data_norm, test_labels, verbose=0)[-1])
                n_model_BAs.append(balance_accuracy(test_labels, test_pred))
                n_model_mAPs.append(average_precision_score(test_labels[:, -1], test_pred))
                train_acc = model.evaluate(train_data_norm, train_labels, verbose=0)[-1]
                print('n_features : ', n_features,
                      ', accuracy : ', n_model_accuracies[-1],
                      ', BA : ', n_model_BAs[-1],
                      ', mAP : ', n_model_mAPs[-1],
                      ', train_accuracy : ', train_acc,
                      ', time : ', n_time[-1], 's')

            heatmap = np.mean(heatmaps, axis=0)
            best_features = np.argsort(heatmap)[::-1][:n_features]

            svc_train_data = train_data[:, best_features]
            svc_test_data = test_data[:, best_features]

            norm = normalization_func()
            svc_train_data_norm = norm.fit_transform(svc_train_data)
            svc_test_data_norm = norm.transform(svc_test_data)

            bestcv = -1
            bestc = None
            bestSolver = None
            for s in [0, 1, 2, 3]:
                for my_c in [0.001, 0.1, 0.5, 1.0, 1.4, 1.5, 1.6, 2.0, 2.5, 5.0, 100.0]:
                    cmd = '-v 5 -s ' + str(s) + ' -c ' + str(my_c) + ' -q'
                    cv = liblinearutil.train((2 * train_labels[:, -1] - 1).tolist(), svc_train_data_norm.tolist(), cmd)
                    if cv > bestcv:
                        # print('Best -> C:', my_c, ', s:', s, ', acc:', cv)
                        bestcv = cv
                        bestc = my_c
                        bestSolver = s
            svc_kwargs['C'] = bestc
            svc_kwargs['solver'] = bestSolver
            print('Best -> C:', bestc, ', s:', bestSolver, ', acc:', bestcv)

            for r in range(reps):
                np.random.seed(cont_seed)
                K.tf.set_random_seed(cont_seed)
                cont_seed += 1

                model = train_SVC(svc_train_data_norm, train_labels, svc_kwargs)
                _, accuracy, test_pred = liblinearutil.predict(
                    (2 * test_labels[:, -1] - 1).tolist(), svc_test_data_norm.tolist(), model, '-q'
                )
                test_pred = np.asarray(test_pred)
                n_svc_accuracies.append(accuracy[0])
                n_svc_BAs.append(balance_accuracy(test_labels, test_pred))
                n_svc_mAPs.append(average_precision_score(test_labels[:, -1], test_pred))
                del model
                model = train_Keras(svc_train_data, train_labels, svc_test_data, test_labels, model_kwargs)
                train_data_norm = model.normalization.transform(svc_train_data)
                test_data_norm = model.normalization.transform(svc_test_data)
                test_pred = model.predict(test_data_norm)
                n_BAs.append(balance_accuracy(test_labels, test_pred))
                n_mAPs.append(average_precision_score(test_labels[:, -1], test_pred))
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
                    ', svc_acc : ', n_svc_accuracies[-1],
                    ', svc_BA : ', n_svc_BAs[-1],
                    ', svc_mAP : ', n_svc_mAPs[-1],
                )
            if i >= len(accuracies):
                accuracies.append(n_accuracies)
                svc_accuracies.append(n_svc_accuracies)
                model_accuracies.append(n_model_accuracies)
                BAs.append(n_BAs)
                mAPs.append(n_mAPs)
                fs_time.append(n_time)
                svc_BAs.append(n_svc_BAs)
                svc_mAPs.append(n_svc_mAPs)
                model_BAs.append(n_model_BAs)
                model_mAPs.append(n_model_mAPs)
                nfeats.append(n_features)
                mus.append(model_kwargs['mu'])
            else:
                accuracies[i] += n_accuracies
                svc_accuracies[i] += n_svc_accuracies
                model_accuracies[i] += n_model_accuracies
                fs_time[i] += n_time
                BAs[i] += n_BAs
                mAPs[i] += n_mAPs
                svc_BAs[i] += n_svc_BAs
                svc_mAPs[i] += n_svc_mAPs
                model_BAs[i] += n_model_BAs
                model_mAPs[i] += n_model_mAPs

    output_filename = directory + 'LinearSVC_' + kernel + '_CAEv1.json'

    if not os.path.isdir(directory):
        os.makedirs(directory)

    info_data = {
        'kernel': kernel,
        'reps': reps,
        'classification': {
            'mus': mus,
            'n_features': nfeats,
            'accuracy': accuracies,
            'mean_accuracy': np.array(accuracies).mean(axis=1).tolist(),
            'svc_accuracy': svc_accuracies,
            'mean_svc_accuracy': np.array(svc_accuracies).mean(axis=1).tolist(),
            'model_accuracy': model_accuracies,
            'mean_model_accuracy': np.array(model_accuracies).mean(axis=1).tolist(),
            'BA': BAs,
            'mean_BA': np.array(BAs).mean(axis=1).tolist(),
            'mAP': mAPs,
            'mean_mAP': np.array(mAPs).mean(axis=1).tolist(),
            'svc_BA': svc_BAs,
            'svc_mean_BA': np.array(svc_BAs).mean(axis=1).tolist(),
            'svc_mAP': svc_mAPs,
            'svc_mean_mAP': np.array(svc_mAPs).mean(axis=1).tolist(),
            'model_BA': model_BAs,
            'model_mean_BA': np.array(model_BAs).mean(axis=1).tolist(),
            'model_mAP': model_mAPs,
            'model_mean_mAP': np.array(model_mAPs).mean(axis=1).tolist(),
            'fs_time': fs_time
        }
    }

    for k, v in info_data['classification'].items():
        if 'mean' in k:
            print(k, v)

    with open(output_filename, 'w') as outfile:
        json.dump(info_data, outfile)


if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../../../')
    main(dataset_name)
