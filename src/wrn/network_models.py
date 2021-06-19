from tensorflow.keras.models import Model
from tensorflow.keras import backend as K, optimizers, layers, models
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input, Convolution2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import EfficientNetB0, DenseNet121, MobileNetV2
from src.wrn.wide_residual_network import wrn_block
from src.network_models import three_layer_nn as tln
import numpy as np
import tempfile
import os


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
        data_augmentation=True, regularization=None
):
    num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
    num_filters_bef_dense_block = 2 * growth_rate

    # start model definition
    # densenet CNNs (composite function) are made of BN-ReLU-Conv2D
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters_bef_dense_block,
                      kernel_size=3, kernel_regularizer=l2(regularization) if regularization > 0.0 else None,
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
                              kernel_size=1, kernel_regularizer=l2(regularization) if regularization > 0.0 else None,
                              padding='same',
                              kernel_initializer='he_normal')(y)
            if not data_augmentation:
                y = layers.Dropout(0.2)(y)
            y = layers.BatchNormalization()(y)
            y = layers.Activation('relu')(y)
            y = layers.Conv2D(growth_rate,
                              kernel_size=3, kernel_regularizer=l2(regularization) if regularization > 0.0 else None,
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
                          kernel_size=1, kernel_regularizer=l2(regularization) if regularization > 0.0 else None,
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
                           kernel_regularizer=l2(regularization) if regularization > 0.0 else None,
                           activation='softmax')(y)

    # instantiate and compile model
    # orig paper uses SGD but RMSprop works better for DenseNet
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(1e-3),
                  metrics=['acc'])

    return model


def efficientnetB0(
        input_shape, nclasses=2, num_dense_blocks=3, growth_rate=12, depth=100, compression_factor=0.5,
        data_augmentation=True, regularization=0.
):

    keras_shape = input_shape
    if input_shape[-1] == 1:
        keras_shape = (32, 32, 3)

    keras_model = EfficientNetB0(
        include_top=False,
        input_shape=keras_shape,
        weights='imagenet'
    )

    keras_model.trainable = True

    # adding regularization
    regularizer = l2(regularization)

    for layer in keras_model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    keras_model.save_weights(tmp_weights_path)

    keras_json = keras_model.to_json()
    keras_model = models.model_from_json(keras_json)
    keras_model.load_weights(tmp_weights_path, by_name=True)

    outputs = keras_model.output
    inputs = keras_model.input
    if input_shape[-1] == 1:
        inputs = layers.Input(shape=input_shape)
        x = layers.ZeroPadding2D(padding=(2,2))(inputs)
        output_shape = K.int_shape(x)
        output_shape = output_shape[:-1] + (3,)
        x = layers.Lambda(lambda x: K.tile(x, (1, 1, 1, 3)), output_shape=output_shape)(x)
        outputs = keras_model(x)

    outputs = layers.Flatten()(outputs)
    # outputs = layers.GlobalAveragePooling2D()(outputs)
    # outputs = layers.Dropout(rate=.5)(outputs)
    outputs = layers.Dense(nclasses,
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(regularization) if regularization > 0.0 else None,
                           activation='softmax')(outputs)

    # instantiate and compile model
    # orig paper uses SGD but RMSprop works better for DenseNet
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(1e-4),
                  metrics=['acc'])

    return model

