from keras.callbacks import Callback
from keras import backend as K


class E2EFSCallback(Callback):

    def __init__(self, factor_func=None, units_func=None, units=0, verbose=0, early_stop=True):
        super(E2EFSCallback, self).__init__()
        self.factor_func = factor_func
        self.units_func = units_func
        self.verbose = verbose
        self.units = units
        self.early_stop = early_stop

    def on_epoch_begin(self, epoch, logs=None):
        layer = self.model.layers[1]
        nnz_units = (K.eval(layer.e2efs_kernel) > 0).sum()
        if hasattr(layer, 'moving_factor') and self.factor_func is not None:
            factor = self.factor_func(epoch)
            if hasattr(layer, 'regularization_loss') and nnz_units <= layer.units:
                factor = 1., 1., 0.
            K.set_value(layer.moving_factor, factor)
        if hasattr(layer, 'moving_units') and self.units_func is not None:
            K.set_value(layer.moving_units, self.units_func(epoch))

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.layers[1]
        moving_factor = K.eval(layer.moving_factor) if hasattr(layer, 'moving_factor') else None
        e2efs_kernel = K.eval(layer.e2efs_kernel)
        if self.verbose > 0 or (e2efs_kernel > 0.).sum() <= self.units or epoch % 100 == 0:
            print(
                "Epoch %05d: cost stopping %.6f" % (epoch, logs['loss']),
                ', moving_factor : ', moving_factor,
                ', moving_units : ', K.eval(layer.moving_units),
                ', nnz : ', (e2efs_kernel > 0.).sum(),
                ', zeros : ', (e2efs_kernel == 0.).sum(),
                ', T : ', K.eval(layer.moving_T),
                ', sum_gamma : ', e2efs_kernel.sum(),
                ', max_gamma : ', e2efs_kernel.max()
            )
        if self.early_stop and (e2efs_kernel > 0.).sum() <= self.units:
            self.model.stop_training = True
            print('Early stopping')
