from keras.layers import Add, Activation, Dropout
from keras.layers import Convolution2D
from keras.layers import BatchNormalization
from keras import backend as K
from keras.regularizers import l2
import numpy as np


def residual_block(ip, output_channels, strides=(1, 1), dropout=0.0, regularization=0.0, kernel_initializer='he_normal', bn=True):
    ip_channels = K.int_shape(ip)[-1]
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    if output_channels == ip_channels and np.prod(strides) == 1.0:
        init = ip
        x = ip
        if bn:
            x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=1e-5, gamma_initializer='ones')(x)
        x = Activation('relu')(x)
    else:
        x = ip
        if bn:
            x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=1e-5, gamma_initializer='ones')(x)
        x = Activation('relu')(x)
        init = Convolution2D(
            output_channels, (1, 1), padding='same', strides=strides, kernel_initializer=kernel_initializer,
            use_bias=False, kernel_regularizer=l2(regularization) if regularization > 0.0 else None
        )(x)

    x = Convolution2D(output_channels, (3, 3), padding='same', strides=strides, kernel_initializer=kernel_initializer,
                      use_bias=False, kernel_regularizer=l2(regularization) if regularization > 0.0 else None)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=1e-5, gamma_initializer='ones')(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Convolution2D(output_channels, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                      use_bias=False, kernel_regularizer=l2(regularization) if regularization > 0.0 else None)(x)

    m = Add()([init, x])

    return m


def wrn_block(ip, output_channels, N, strides=(1, 1), dropout=0.0, regularization=0.0, kernel_initializer='he_normal', bn=True):
    m = residual_block(
        ip, output_channels, strides, dropout, regularization=regularization, kernel_initializer=kernel_initializer, bn=bn
    )
    for i in range(N-1):
        m = residual_block(
            m, output_channels, dropout=dropout, regularization=regularization, kernel_initializer=kernel_initializer, bn=bn
        )
    return m


def create_cifar_wide_residual_network(
        ip, l=16, k=1, dropout=0.3, regularization=1e-5, kernel_initializer='he_normal'
):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # ip = Input(shape=input_dim)

    x = Convolution2D(
        16, (3, 3), padding='same', kernel_initializer=kernel_initializer,
        use_bias=False, kernel_regularizer=l2(regularization) if regularization > 0.0 else None
    )(ip)

    output_channel_basis = [16, 32, 64]
    strides = [1, 2, 2]

    N = (l - 4) // 6

    for ocb, stride in zip(output_channel_basis, strides):
        x = wrn_block(
            x, ocb * k, N, strides=(stride, stride), dropout=dropout,
            regularization=regularization, kernel_initializer=kernel_initializer
        )

    x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=1e-5, gamma_initializer='ones')(x)
    x = Activation('relu')(x)

    return x


def create_stl_wide_residual_network(
        ip, l=16, k=1, dropout=0.3, regularization=1e-5, kernel_initializer='he_normal'
):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # ip = Input(shape=input_dim)

    x = Convolution2D(
        16, (3, 3), strides=(3, 3), padding='same', kernel_initializer=kernel_initializer,
        use_bias=False, kernel_regularizer=l2(regularization) if regularization > 0.0 else None
    )(ip)

    output_channel_basis = [16, 32, 64]
    strides = [1, 2, 2]

    N = (l - 4) // 6

    for ocb, stride in zip(output_channel_basis, strides):
        x = wrn_block(
            x, ocb * k, N, strides=(stride, stride), dropout=dropout,
            regularization=regularization, kernel_initializer=kernel_initializer
        )

    x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=1e-5, gamma_initializer='ones')(x)
    x = Activation('relu')(x)

    return x
