from keras import backend as K, regularizers
from sklearn.svm import LinearSVC as sklearn_LinearSVC
from keras.layers import Dense, Input
from keras.models import Model
from keras import optimizers
import numpy as np


class LinearSVC(object):
    def __init__(
            self, nfeatures, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True,
            probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
            decision_function_shape='ovr', random_state=None, use_keras=True, keras_class_weight=None, mu=1e-3
    ):
        self.nfeatures = nfeatures
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        self.use_keras = use_keras
        self.keras_class_weight = keras_class_weight
        self.mu = mu

    def fit(self, X, y, sample_weight=None):
        if isinstance(self.gamma, str):
            if 'auto' in self.gamma:
                self.gamma = 1. / self.nfeatures[0]
            elif self.gamma == 'scale':
                self.gamma = 1. / (self.nfeatures[0] * X.std())
            else:
                raise Exception('gamma ' + self.gamma + ' not supported')
        if isinstance(self.class_weight, str):
            if self.class_weight == 'balanced':
                self.class_weight_keras = y.shape[0] / (y.shape[1] * np.sum(y, axis=0))
            else:
                raise Exception('class_weight ' + self.gamma + ' not supported')
        self.sklearn_model = sklearn_LinearSVC(
            C=self.C, tol=self.tol, class_weight=self.class_weight, verbose=self.verbose, max_iter=self.max_iter,
            random_state=self.random_state
        )
        arg_y = 2 * np.argmax(y, axis=-1) - 1
        self.sklearn_model.fit(X, arg_y, sample_weight=sample_weight)
        if self.use_keras:
            self.create_keras_model()

    def create_keras_model(self, nclasses, warming_up=False):
        input = Input(shape=self.nfeatures)
        x = input
        classifier = Dense(
            nclasses - 1, use_bias=True, kernel_initializer='zeros',
            bias_initializer='zeros', input_shape=K.int_shape(x)[-1:],
            kernel_regularizer=regularizers.l2(self.mu)
        )
        output = classifier(x)
        if warming_up:
            kernel_values = self.sklearn_model.coef_.T
            kernel_values = np.concatenate((np.mean(-1. * kernel_values, axis=-1)[:, None], kernel_values), axis=-1)
            beta = self.sklearn_model.intercept_
            classifier.set_weights([kernel_values, beta])
        self.model = Model(input, output)
        self.output_shape = self.model.output_shape
        self.output = self.model.output
        self.layers = self.model.layers
        self.input = self.model.input
        optimizer = optimizers.SGD(0.1)
        self.model.compile(loss=self.loss_function('square_hinge'), optimizer=optimizer, metrics=['acc'])

    @classmethod
    def loss_function(cls, loss_function, weights=None):
        def loss(y_true, y_pred):
            y_true = 2. * y_true[:, 1:] - 1.
            out = K.relu(1.0 - y_true * y_pred)
            if 'square' in loss_function:
                out = K.square(out)
            if weights is not None:
                out = weights * out
            return K.mean(out, axis=-1)
        return loss

    @classmethod
    def accuracy(cls, y_true, y_pred):
        y_true = 2. * y_true[:, 1:] - 1.
        return K.mean(K.equal(y_true, K.sign(y_pred)))

    @classmethod
    def mAP(cls, y_true, y_pred):
        y_pred_index = K.flatten(K.cast(K.relu(K.sign(y_pred)), 'int32'))
        y_pred_one_hot = K.one_hot(y_pred_index, 2)
        return K.mean(K.sum(y_true * y_pred_one_hot, axis=0) / K.maximum(1., K.sum(y_true, axis=0)))

    def evaluate(self, X, y, batch_size=None, verbose=1):
        if self.use_keras:
            model_eval = self.model.evaluate(X, y, batch_size=batch_size, verbose=1)
        else:
            y = 2. * y[:, 1:] - 1.
            model_eval = [self.sklearn_model.score(X, y)]
        return model_eval

    def predict(self, X, batch_size=None, verbose=1):
        if self.use_keras:
            predict = self.model.predict(X, batch_size=batch_size, verbose=verbose)
        else:
            predict = self.sklearn_model.predict(X)
        return predict

    def save_keras_model(self, filename):
        self.model.save(filename)

    def __del__(self):
        try:
            del self.model
        except:
            pass
        try:
            del self.sklearn_model
        except:
            pass
