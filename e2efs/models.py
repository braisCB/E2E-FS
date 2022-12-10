from e2efs import callbacks as custom_callbacks
from keras import backend as K
import tensorflow as tf
import numpy as np
from packaging import version
if version.parse(tf.__version__) < version.parse('2.0'):
    from e2efs import optimizers as custom_optimizers
    from e2efs import e2efs_layers
else:
    if version.parse(tf.__version__) < version.parse('2.9'):
        from e2efs import optimizers_tf2 as custom_optimizers
    else:
        from e2efs import optimizers_tf29 as custom_optimizers
    from e2efs import e2efs_layers_tf2 as e2efs_layers_layers


class E2EFSBase:

    def __init__(self, th=0.1):
        self.th = th
        self.model = None
        self.e2efs_layer = None

    def get_layer(self, input_shape):
        raise NotImplementedError

    def attach(self, model):
        self.e2efs_layer = self.get_layer(model.input_shape[1:])
        self.model = self.e2efs_layer.add_to_model(model, input_shape=model.input_shape[1:])
        kwargs = model.optimizer.get_config()
        if 'sgd' in type(model.optimizer).__name__.lower():
            opt = custom_optimizers.E2EFS_SGD(self.e2efs_layer, th=self.th, **kwargs)
        elif 'adam' in type(model.optimizer).__name__.lower():
            opt = custom_optimizers.E2EFS_Adam(self.e2efs_layer, th=self.th, **kwargs)
        elif 'RMSprop' in type(model.optimizer).__name__.lower():
            opt = custom_optimizers.E2EFS_RMSprop(self.e2efs_layer, th=self.th, **kwargs)
        else:
            raise Exception('Optimizer not supported. Contact the authors if you need it')
        compile_args = model._get_compile_args()
        compile_args['optimizer'] = opt
        self.model.compile(**compile_args)
        return self

    def fit(self, x,
            y,
            epochs=1000000,
            batch_size=None,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            **kwargs):

        callbacks = callbacks or list()
        callbacks.append(custom_callbacks.E2EFSCallback(verbose=verbose))
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks, validation_split=validation_split,
                       validation_data=validation_data, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight,
                       initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, **kwargs)
        return self

    def fine_tuning(self, x,
                    y,
                    epochs=1,
                    batch_size=None,
                    verbose=1,
                    callbacks=None,
                    validation_split=0.,
                    validation_data=None,
                    shuffle=True,
                    class_weight=None,
                    sample_weight=None,
                    initial_epoch=0,
                    steps_per_epoch=None,
                    validation_steps=None,
                    **kwargs):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks,
                       validation_split=validation_split,
                       validation_data=validation_data, shuffle=shuffle, class_weight=class_weight,
                       sample_weight=sample_weight,
                       initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                       **kwargs)
        return self

    def get_mask(self):
        self._check_model()
        input_shape = self.model.input_shape[1:]
        return K.eval(self.model.fs_kernel()).reshape(input_shape)

    def get_heatmap(self):
        self._check_model()
        input_shape = self.model.input_shape[1:]
        return K.eval(self.model.heatmap).reshape(input_shape)

    def get_ranking(self):
        self._check_model()
        input_shape = self.model.input_shape[1:]
        return np.argsort(-1. * K.eval(self.model.heatmap)).reshape(input_shape)

    def _check_model(self):
        if self.model is None:
            raise Exception('Model not trained. Did you call fit first?')

    def get_model(self):
        self._check_model()
        return self.model


class E2EFSSoft(E2EFSBase):
    
    def __init__(self, n_features_to_select, rho=0.25, T=10000, warmup_T=2000, th=.1, alpha_M=.99, epsilon=.001):
        self.n_features_to_select = n_features_to_select
        self.rho = rho
        self.T = T
        self.warmup_T = warmup_T
        self.alpha_M = alpha_M
        self.epsilon = epsilon
        super(E2EFSSoft, self).__init__(th)

    def get_layer(self, input_shape):
        return e2efs_layers.E2EFSSoft(self.n_features_to_select, T=self.T, warmup_T=self.warmup_T, decay_factor=1. - self.rho,
                               alpha_N=self.alpha_M, epsilon=self.epsilon, input_shape=input_shape)
    

class E2EFS(E2EFSSoft):

    def __init__(self, n_features_to_select, T=10000, warmup_T=2000, th=.1, alpha_M=.99, epsilon=.001):
        rho=1.
        super(E2EFS, self).__init__(n_features_to_select, rho, T, warmup_T, th, alpha_M, epsilon)


class E2EFSRanking(E2EFSBase):

    def __init__(self, T=20000, warmup_T=2000, th=.1, alpha_M=.99, tau=4.):
        self.n_features_to_select = 1
        self.T = T
        self.warmup_T = warmup_T
        self.alpha_M = alpha_M
        self.tau = tau
        super(E2EFSRanking, self).__init__(th)

    def get_layer(self, input_shape):
        return e2efs_layers.E2EFSRanking(self.n_features_to_select, speedup=self.tau, input_shape=input_shape)

