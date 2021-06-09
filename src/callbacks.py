from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


class E2EFSCallback(Callback):

    def __init__(self, units=None, verbose=0, early_stop=True):
        super(E2EFSCallback, self).__init__()
        self.verbose = verbose
        self.units = units
        self.early_stop = early_stop

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.layers[1]
        units = self.units or layer.units
        moving_factor = K.eval(layer.moving_factor) if hasattr(layer, 'moving_factor') else None
        e2efs_kernel = K.eval(layer.e2efs_kernel())
        if self.verbose > 0 or (e2efs_kernel > 0.).sum() <= units or epoch % 100 == 0:
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
        if self.early_stop and (e2efs_kernel > 0.).sum() <= units:
            self.model.stop_training = True
            print('Early stopping')
