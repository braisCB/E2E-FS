from keras.models import Model
from keras import backend as K, optimizers
from keras.layers import Dense, Activation, BatchNormalization, Input, Convolution2D, GlobalAveragePooling2D, Flatten
from keras.regularizers import l2
from src.wrn.wide_residual_network import wrn_block
from src.network_models import three_layer_nn as tln
import numpy as np


def three_layer_nn(input_shape, nclasses=2, bn=True, kernel_initializer='he_normal',
                   dropout=0.0, lasso=0.0, regularization=5e-4, momentum=0.9):

    nfeatures = np.prod(input_shape)
    tln_model = tln((nfeatures, ), nclasses, bn, kernel_initializer, dropout, lasso, regularization, momentum)
    ip = Input(shape=input_shape)
    x = ip
    if len(input_shape) > 1:
        x = Flatten()(x)
    output = tln_model(x)
    model = Model(ip, output)

    optimizer = optimizers.SGD(lr=1e-1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model



def wrn164(
    input_shape, nclasses=2, bn=True, kernel_initializer='he_normal', dropout=0.0, regularization=0.0,
    softmax=True
):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    ip = Input(shape=input_shape)

    x = ip

    x = Convolution2D(
        16, (3, 3), padding='same', kernel_initializer=kernel_initializer,
        use_bias=False, kernel_regularizer=l2(regularization) if regularization > 0.0 else None
    )(x)

    l = 16
    k = 4

    output_channel_basis = [16, 32, 64]
    strides = [1, 2, 2]

    N = (l - 4) // 6

    for ocb, stride in zip(output_channel_basis, strides):
        x = wrn_block(
            x, ocb * k, N, strides=(stride, stride), dropout=dropout,
            regularization=regularization, kernel_initializer=kernel_initializer, bn=bn
        )

    if bn:
        x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=1e-5, gamma_initializer='ones')(x)
    x = Activation('relu')(x)

    deep_features = GlobalAveragePooling2D()(x)

    classifier = Dense(nclasses, kernel_initializer=kernel_initializer)

    output = classifier(deep_features)
    if softmax:
        output = Activation('softmax')(output)

    model = Model(ip, output)

    optimizer = optimizers.SGD(lr=1e-1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

