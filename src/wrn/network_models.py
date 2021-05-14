from keras.models import Model
from keras import backend as K, optimizers, layers, models
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


def densenet(
        input_shape, nclasses=2, num_dense_blocks=3, growth_rate=12, depth=100, compression_factor=0.5,
        data_augmentation=True
):
    num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
    num_filters_bef_dense_block = 2 * growth_rate

    # start model definition
    # densenet CNNs (composite function) are made of BN-ReLU-Conv2D
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters_bef_dense_block,
                      kernel_size=3,
                      padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.concatenate([inputs, x])

    # stack of dense blocks bridged by transition layers
    for i in range(num_dense_blocks):
        # a dense block is a stack of bottleneck layers
        for j in range(num_bottleneck_layers):
            y = layers.BatchNormalization()(x)
            y = layers.Activation('relu')(y)
            y = layers.Conv2D(4 * growth_rate,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer='he_normal')(y)
            if not data_augmentation:
                y = layers.Dropout(0.2)(y)
            y = layers.BatchNormalization()(y)
            y = layers.Activation('relu')(y)
            y = layers.Conv2D(growth_rate,
                              kernel_size=3,
                              padding='same',
                              kernel_initializer='he_normal')(y)
            if not data_augmentation:
                y = layers.Dropout(0.2)(y)
            x = layers.concatenate([x, y])

        # no transition layer after the last dense block
        if i == num_dense_blocks - 1:
            continue

        # transition layer compresses num of feature maps and reduces the size by 2
        num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
        num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
        y = layers.BatchNormalization()(x)
        y = layers.Conv2D(num_filters_bef_dense_block,
                          kernel_size=1,
                          padding='same',
                          kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y = layers.Dropout(0.2)(y)
        x = layers.AveragePooling2D()(y)

    # add classifier on top
    # after average pooling, size of feature map is 1 x 1
    x = layers.AveragePooling2D()(x)
    y = layers.Flatten()(x)
    outputs = layers.Dense(nclasses,
                           kernel_initializer='he_normal',
                           activation='softmax')(y)

    # instantiate and compile model
    # orig paper uses SGD but RMSprop works better for DenseNet
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(1e-3),
                  metrics=['acc'])

    return model

