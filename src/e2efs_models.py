from typing import Optional, Any

import lightning as pl
import torch
from .layers import E2EFSMaskBase


class E2EFSModel(pl.LightningModule):

    def __init__(self, network: pl.LightningModule, e2efs_layer: E2EFSMaskBase,
                 epsilon: float = 1e-8, threshold: float = 1.):
        super(E2EFSModel, self).__init__()
        self.network = network
        self.e2efs_layer = e2efs_layer
        self.epsilon = epsilon
        self.register_buffer('threshold', torch.tensor(threshold))
        self.automatic_optimization = False
        self.network_log = self.network.log
        self.fitted = False

    def forward(self, x):
        masked_x = self.e2efs_layer(x)
        output = self.network.forward(masked_x)
        return output

    def __combine_grads(self, network_grad, e2efs_grad):
        alpha = self.e2efs_layer.moving_factor
        network_grad_norm = torch.norm(network_grad) + self.epsilon
        e2efs_grad_normalized = e2efs_grad / (torch.norm(e2efs_grad) + self.epsilon)
        network_grad_normalized = network_grad / network_grad_norm
        combined_e2efs_grad = alpha * e2efs_grad_normalized + (1. - alpha) * network_grad_normalized
        combined_e2efs_grad = (torch.sign(alpha) *
            torch.maximum(self.threshold, network_grad_norm) *
            combined_e2efs_grad / (torch.norm(combined_e2efs_grad) + self.epsilon))
        return combined_e2efs_grad

    def training_step(self, batch, batch_idx):
        self.network.log = self.log
        e2efs_opt, network_opt = self.optimizers()
        x, y, sample_weights = batch
        masked_x = self.e2efs_layer(x)
        if self.fitted:
            masked_x = masked_x.detach()
        network_loss = self.network.training_step(batch=(masked_x, y, sample_weights), batch_idx=batch_idx)
        e2efs_opt.zero_grad()
        network_opt.zero_grad()
        self.manual_backward(network_loss)
        if not self.fitted:
            network_grad = self.e2efs_layer.kernel.grad
            e2efs_opt.zero_grad()
            penalty = self.e2efs_layer.get_penalty()
            self.manual_backward(penalty)
            penalty_grad = self.e2efs_layer.kernel.grad
            self.e2efs_layer.kernel.grad = self.__combine_grads(network_grad, penalty_grad)
            e2efs_opt.step()
            self.log_dict({"alpha": self.e2efs_layer.moving_factor, "penalty": penalty}, prog_bar=True)
        network_opt.step()
        self.e2efs_layer.kernel_constraint()
        # loss = F.cross_entropy(y_hat, y) # - 1e-4*torch.sum(s * torch.log(s_clip))
        nfeats = self.e2efs_layer.get_n_alive().float()
        self.log_dict({"nfeats": nfeats}, prog_bar=True)
        self.e2efs_layer.update_buffers()
        self.network.log = self.network_log

    def validation_step(self, batch, batch_idx):
        self.network.log = self.log
        x, y = batch
        masked_x = self.e2efs_layer(x)
        self.network.validation_step(batch=(masked_x, y), batch_idx=batch_idx)
        self.network.log = self.network_log

    def test_step(self, batch, batch_idx):
        self.network.log = self.log
        x, y = batch
        masked_x = self.e2efs_layer(x)
        self.network.test_step(batch=(masked_x, y), batch_idx=batch_idx)
        self.network.log = self.network_log

    def configure_optimizers(self):
        network_opt = self.network.configure_optimizers()
        e2efs_opt = torch.optim.Adam(self.e2efs_layer.parameters(), lr=1e-3 if self.fitted else 1e-3)
        # e2efs_opt = torch.optim.SGD(self.e2efs_layer.parameters(), lr=.01)
        callbacks = []
        if self.fitted:
            callbacks.append(
                torch.optim.lr_scheduler.StepLR(network_opt, 160, gamma=.2)
            )
        return [e2efs_opt, network_opt], callbacks

    def on_train_epoch_end(self):
        self.network.log = self.log
        self.network.on_train_epoch_end()
        self.network.log = self.network_log
        nfeats = self.e2efs_layer.get_n_alive().float()
        if nfeats <= self.e2efs_layer.n_features_to_select:
            self.fitted = True

    def on_validation_epoch_end(self):
        self.network.log = self.log
        self.network.on_validation_epoch_end()
        self.network.log = self.network_log

    def on_test_epoch_end(self) -> None:
        self.network.log = self.log
        output = self.network.on_test_epoch_end()
        self.network.log = self.network_log
        return output

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self.forward(batch[0])

