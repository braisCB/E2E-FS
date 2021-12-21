from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks, regularizers, layers, models, optimizers
import json
import numpy as np
import os
from dataset_reader import redundancy_1
from src.network_models import three_layer_nn
from sklearn.model_selection import RepeatedStratifiedKFold
from tensorflow.keras import backend as K
import time
import tensorflow as tf


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
normalization_func = redundancy_1.Normalize

dataset_name = 'redundancy_1'
directory = os.path.dirname(os.path.realpath(__file__)) + '/info/'

initial_lr = .001


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

    net = layers.Dense(100, activation=activation, name='s/dense1',
                kernel_regularizer=regularizers.l2(1e-3))(model_input)
    net = layers.Dense(100, activation=activation, name='s/dense2',
                kernel_regularizer=regularizers.l2(1e-3))(net)

    # A tensor of shape, [batch_size, max_sents, 100]
    logits = layers.Dense(np.prod(input_shape))(net)
    # [BATCH_SIZE, max_sents, 1]
    tau = 0.1
    samples = Sample_Concrete(tau, nfeatures, name='sample')(logits)
    return models.Model(model_input, samples)


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
    dataset = redundancy_1.load_dataset()
    return dataset


def train_Keras(train_X, train_y, test_X, test_y, kwargs, l2x_model_func=None, n_features=None, epochs=150):
    normalization = normalization_func()
    num_classes = train_y.shape[-1]

    norm_train_X = normalization.fit_transform(train_X)
    norm_test_X = normalization.transform(test_X)

    batch_size = max(2, len(train_X) // 50)
    class_weight = train_y.shape[0] / np.sum(train_y, axis=0)
    class_weight = num_classes * class_weight / class_weight.sum()
    sample_weight = class_weight[np.argmax(train_y, axis=1)]
    class_weight = None

    classifier = three_layer_nn(nfeatures=norm_train_X.shape[1:], nclasses=num_classes, **kwargs)

    model_clbks = [
        callbacks.LearningRateScheduler(scheduler()),
    ]

    fs_callbacks = [
        callbacks.LearningRateScheduler(scheduler(extra_epochs=extra_epochs)),
    ]

    l2x_model = l2x_model_func(norm_train_X.shape[1:], n_features)
    classifier_input = layers.Multiply()([l2x_model.output, l2x_model.input])
    output = classifier(classifier_input)
    model = models.Model(l2x_model.input, output)

    optimizer = optimizer_class(lr=initial_lr)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['acc']
    )

    model.l2x_model = l2x_model

    start_time = time.process_time()
    model.fit(
        norm_train_X, train_y, batch_size=batch_size,
        epochs=epochs + extra_epochs,
        callbacks=fs_callbacks,
        validation_data=(norm_test_X, test_y),
        class_weight=class_weight,
        sample_weight=sample_weight,
        verbose=0
    )
    # scores = l2x_model.predict(norm_train_X, verbose=0, batch_size=batch_size).reshape((-1, np.prod(norm_train_X.shape[1:])))
    # model.heatmap = compute_median_rank(scores, k=n_features)
    model.heatmap = l2x_model.predict(norm_train_X, verbose=0, batch_size=batch_size).reshape((-1, np.prod(norm_train_X.shape[1:]))).sum(axis=0)
    model.fs_time = time.process_time() - start_time

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
    fs_time = []
    real_feats = []
    redundant_feats = []
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

        model_kwargs = {}

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
                    l2x_model_func=get_l2x_model, n_features=n_features,
                )
                heatmaps.append(model.heatmap)
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

    output_filename = directory + 'three_layer_nn_L2X.json'

    if not os.path.isdir(directory):
        os.makedirs(directory)

    info_data = {
        'kernel': kernel,
        'reps': reps,
        'classification': {
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
