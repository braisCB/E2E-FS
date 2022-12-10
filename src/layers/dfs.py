from keras import backend as K, layers, models, initializers, regularizers, constraints
import numpy as np


class DFS(layers.Layer):

    def __init__(self,
                 kernel_initializer='ones',
                 kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3),
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DFS, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        input_dim = input_shape[1:]
        self.kernel = self.add_weight(shape=input_dim,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.built = True

    def call(self, inputs, **kwargs):
        output = inputs * self.kernel
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(DFS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def add_to_model(self, model, input_shape, activation=None):
        input = layers.Input(shape=input_shape)
        x = self(input)
        if activation is not None:
            x = layers.Activation(activation=activation)(x)
        output = model(x)
        model = models.Model(input, output)
        model.fs_kernel = K.abs(self.kernel)
        return model


    def compute_output_shape(self, input_shape):
        return input_shape

