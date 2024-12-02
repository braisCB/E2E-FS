import lightning as pl
import torch


class MyEarlyStopping(pl.pytorch.callbacks.EarlyStopping):

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer, pl_module)

    def _run_early_stopping_check(self, trainer, pl_module):
        nfeats = pl_module.e2efs_layer.get_n_alive().item()
        if nfeats < self.stopping_threshold:
            trainer.should_stop = True
        print('nfeats {} (threshold {})'.format(nfeats, self.stopping_threshold))
