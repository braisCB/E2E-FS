from keras import backend as K, layers, models, initializers, regularizers
import numpy as np


class E2EFS_Base(layers.Layer):

    def __init__(self, units,
                 kernel_initializer='truncated_normal',
                 kernel_constraint=None,
                 kernel_activation=None,
                 kernel_regularizer=None,
                 heatmap_momentum=.99999,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(E2EFS_Base, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_constraint = kernel_constraint
        self.kernel_activation = kernel_activation
        self.kernel_regularizer = kernel_regularizer
        self.supports_masking = True
        self.kernel = None
        self.heatmap_momentum = heatmap_momentum

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = np.prod(input_shape[1:])
        kernel_shape = (input_dim, )
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=self.trainable)
        self.moving_heatmap = self.add_weight(shape=(input_dim, ),
                                              name='heatmap',
                                              initializer='ones',
                                              trainable=False)
        self.e2efs_kernel = self.kernel if self.kernel_activation is None else self.kernel_activation(self.kernel)
        self.stateful = False
        self.built = True

    def call(self, inputs, training=None, **kwargs):

        kernel = self.kernel
        if self.kernel_activation is not None:
            kernel = self.kernel_activation(kernel)
        kernel_clipped = K.reshape(kernel, shape=inputs.shape[1:])

        output = inputs * kernel_clipped

        if training in {0, False}:
            return output

        update_list = self._get_update_list(kernel)
        self.add_update(update_list, inputs)

        return output

    def _get_update_list(self, kernel):

        update_list = [
            K.moving_average_update(self.moving_heatmap, K.sign(kernel), self.heatmap_momentum),
        ]
        return update_list

    def add_to_model(self, model, input_shape, activation=None):
        input = layers.Input(shape=input_shape)
        x = self(input)
        if activation is not None:
            x = layers.Activation(activation=activation)(x)
        output = model(x)
        model = models.Model(input, output)
        model.fs_kernel = self.e2efs_kernel
        model.heatmap = self.moving_heatmap
        return model

    def compute_output_shape(self, input_shape):
        return input_shape


class E2EFSSoft(E2EFS_Base):

    def __init__(self, units,
                 dropout=.0,
                 decay_factor=.5,
                 kernel_regularizer=None,
                 kernel_initializer='ones',
                 T=10000,
                 warmup_T=2000,
                 start_alpha=.0,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.dropout = dropout
        self.decay_factor = decay_factor
        self.T = T
        self.warmup_T = warmup_T
        self.start_alpha = start_alpha
        self.cont_T = 0
        super(E2EFSSoft, self).__init__(units=units,
                                        kernel_regularizer=kernel_regularizer,
                                        kernel_initializer=kernel_initializer,
                                        heatmap_momentum= (T - 1.) / T,
                                        **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        self.moving_units = self.add_weight(shape=(),
                                            name='moving_units',
                                            initializer=initializers.constant(self.units),
                                            trainable=False)
        self.moving_T = self.add_weight(shape=(),
                                        name='moving_T',
                                        initializer='zeros',
                                        trainable=False)
        self.moving_factor = self.add_weight(shape=(),
                                             name='moving_factor',
                                             initializer=initializers.constant([0.]),
                                             trainable=False)
        self.moving_decay = self.add_weight(shape=(),
                                             name='moving_decay',
                                             initializer=initializers.constant(self.decay_factor),
                                             trainable=False)
        self.cont = self.add_weight(shape=(),
                                    name='cont',
                                    initializer='ones',
                                    trainable=False)

        def kernel_activation(x):
            t = x / K.max(K.abs(x))
            s = K.switch(K.less(t, K.epsilon()), K.zeros_like(x), x)
            # s /= K.stop_gradient(K.max(s))
            return s

        self.kernel_activation = kernel_activation

        def kernel_constraint(x):
            return K.clip(x, 0., 1.)

        self.kernel_constraint = kernel_constraint

        def loss_units(x):
            t = x / K.max(K.abs(x))
            x = K.switch(K.less(t, K.epsilon()), K.zeros_like(x), x)
            m = K.sum(K.cast(K.greater(x, 0.), K.floatx()))
            sum_x = K.sum(x)
            moving_units = K.switch(K.less_equal(m, self.units), m,
                                    (1. - self.moving_decay) * self.moving_units)
            epsilon_minus = 0.
            epsilon_plus = K.switch(K.less_equal(m, self.units), self.moving_units, 0.)
            return K.relu(moving_units - sum_x - epsilon_minus) + K.relu(sum_x - moving_units - epsilon_plus)

        # self.kernel_regularizer = lambda x: regularizers.l2(.01)(K.relu(x))

        super(E2EFSSoft, self).build(input_shape)

        def regularization(x):
            l_units = loss_units(x)
            t = x / K.max(K.abs(x))
            p = K.switch(K.less(t, K.epsilon()), K.zeros_like(x), x)
            cost = K.cast_to_floatx(0.)
            cost += K.sum(p * (1. - p)) + 2. * l_units
            # cost += K.sum(K.relu(x - 1.))
            return cost

        self.regularization_loss = regularization(self.kernel)

    def _get_update_list(self, kernel):
        update_list = super(E2EFSSoft, self)._get_update_list(kernel)
        update_list += [
            (self.moving_factor, K.switch(K.less(self.moving_T, self.warmup_T),
                                          self.start_alpha,
                                          K.minimum(.99, self.start_alpha + (1. - self.start_alpha) * (self.moving_T - self.warmup_T) / self.T))),
            (self.moving_T, self.moving_T + 1),
            (self.moving_decay, K.switch(K.less(self.moving_factor, .99), self.moving_decay, K.maximum(.75, self.moving_decay + 1e-3)))
        ]
        return update_list


class E2EFS(E2EFSSoft):

    def __init__(self, units,
                 dropout=.0,
                 kernel_initializer='ones',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(E2EFS, self).__init__(units=units,
                                    dropout=dropout,
                                    kernel_regularizer=None,
                                    decay_factor=0.,
                                    kernel_initializer=kernel_initializer,
                                    T=10000,
                                    **kwargs)


class E2EFSRanking(E2EFS_Base):

    def __init__(self, units,
                 dropout=.0,
                 kernel_regularizer=None,
                 kernel_initializer='ones',
                 T=100000,
                 warmup_T=0,
                 start_alpha=.0,
                 speedup=3.,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.dropout = dropout
        self.T = T
        self.warmup_T = warmup_T
        self.start_alpha = start_alpha
        self.cont_T = 0
        self.speedup = speedup
        super(E2EFSRanking, self).__init__(units=units,
                                        kernel_regularizer=kernel_regularizer,
                                        kernel_initializer=kernel_initializer,
                                        heatmap_momentum= (T - 1.) / T,
                                        **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        self.moving_units = self.add_weight(shape=(),
                                            name='moving_units',
                                            initializer=initializers.constant(self.units),
                                            trainable=False)
        self.moving_T = self.add_weight(shape=(),
                                        name='moving_T',
                                        initializer='zeros',
                                        trainable=False)
        self.moving_factor = self.add_weight(shape=(),
                                             name='moving_factor',
                                             initializer=initializers.constant([0.]),
                                             trainable=False)
        self.cont = self.add_weight(shape=(),
                                    name='cont',
                                    initializer='ones',
                                    trainable=False)

        def apply_dropout(x, rate, refactor=False):
            if 0. < self.dropout < 1.:
                def dropped_inputs():
                    x_shape = K.int_shape(x)
                    noise = K.random_uniform(x_shape)
                    factor = 1. / (1. - rate) if refactor else 1.
                    return K.switch(K.less(noise, self.dropout), K.zeros_like(x), factor * x)
                return K.in_train_phase(dropped_inputs, x)
            return x

        def kernel_activation(x):
            x = apply_dropout(x, self.dropout, False)
            t = x / K.max(K.abs(x))
            s = K.switch(K.less(t, K.epsilon()), K.zeros_like(x), x)
            # s /= K.stop_gradient(K.max(s))
            return s

        self.kernel_activation = kernel_activation

        def kernel_constraint(x):
            return K.clip(x, 0., 1.)

        self.kernel_constraint = kernel_constraint

        def loss_units(x):
            t = x / K.max(K.abs(x))
            x = K.switch(K.less(t, K.epsilon()), K.zeros_like(x), x)
            # m = K.sum(K.cast(K.greater(x, 0.), K.floatx()))
            sum_x = K.sum(x)
            # moving_units = K.switch(K.less_equal(m, self.units), m, self.moving_units)
            # epsilon_minus = 0.
            # epsilon_plus = K.switch(K.less_equal(m, self.units), self.moving_units, 0.)
            return K.abs(self.moving_units - sum_x)

        # self.kernel_regularizer = lambda x: regularizers.l2(.01)(K.relu(x))
        # self.kernel_initializer = initializers.constant(max(.05, self.units / np.prod(input_shape[1:])))

        super(E2EFSRanking, self).build(input_shape)

        def regularization(x):
            l_units = loss_units(x)
            t = x / K.max(K.abs(x))
            p = K.switch(K.less(t, K.epsilon()), K.zeros_like(x), x)
            cost = K.cast_to_floatx(0.)
            cost += K.sum(p) - K.sum(K.square(p)) + 2. * l_units
            # cost += K.sum(p * (1. - p)) + l_units
            # cost += K.sum(K.relu(x - 1.))
            return cost

        self.regularization_loss = regularization(self.kernel)

    def _get_update_list(self, kernel):
        update_list = super(E2EFSRanking, self)._get_update_list(kernel)
        update_list += [
            (self.moving_factor, K.switch(K.less_equal(self.moving_T, self.warmup_T),
                                          self.start_alpha,
                                          K.minimum(1., self.start_alpha + (1. - self.start_alpha) * (self.moving_T - self.warmup_T) / self.T))),
            (self.moving_T, self.moving_T + 1),
            (self.moving_units, K.switch(K.less_equal(self.moving_T, self.warmup_T),
                                         K.cast_to_floatx((1. - self.start_alpha) * np.prod(K.int_shape(kernel))),
                                         K.maximum(1., np.prod(K.int_shape(kernel)) * K.pow(K.cast_to_floatx(1. / np.prod(K.int_shape(kernel))), self.speedup * (self.moving_T - self.warmup_T) / self.T)))),
                                         # K.maximum(1., (self.T - self.start_alpha - self.speedup * (self.moving_T - self.warmup_T)) * np.prod(K.int_shape(kernel)) / self.T))),
        ]
        return update_list
