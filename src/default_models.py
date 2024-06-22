import lightning as pl
import torchmetrics
from .metrics import BalancedAccuracy
from . import networks
from torch import nn
import torch
import numpy as np


class DefaultClassifier(pl.LightningModule):

    def __init__(self, input_shape: int, output_size: int, architecture: 'str' = 'linear', regularization=0.) -> None:
        super(DefaultClassifier, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.architecture = architecture
        self.model = self.__create_model()
        self.train_ba = BalancedAccuracy(num_classes=output_size)
        self.val_ba = BalancedAccuracy(num_classes=output_size)
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_size)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_size)
        #self.loss = nn.CrossEntropyLoss()
        self.loss = self.__hinge_loss
        self.regularization = regularization

    def __hinge_loss(self, y_hat, y, sample_weight=None):
        y_m = (2. * y - 1.).view(-1, 1)
        y_m = torch.cat([-y_m, y_m], dim=1)
        loss = torch.square(torch.relu(1. - y_m * y_hat)).sum(dim=-1)
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()

    def __create_model(self) -> nn.Module:
        if self.architecture == 'linear':
            return networks.LinearModel(np.prod(self.input_shape), self.output_size)
        elif self.architecture == 'three_layer_nn':
            return networks.ThreeLayerNNModel(np.prod(self.input_shape), self.output_size)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        if 'conv' not in self.architecture:
            x = x.view((x.size(0), -1))
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y, sample_weight = batch
        y_hat = self.forward(x)
        acc = self.train_acc(y_hat, y)
        self.train_ba.update(y_hat, y)
        loss = self.loss(y_hat, y, sample_weight=sample_weight) # - 1e-4*torch.sum(s * torch.log(s_clip))
        self.log('train_loss', loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.val_acc.update(y_hat, y)
        self.val_ba.update(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.val_acc.update(y_hat, y)
        self.val_ba.update(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, weight_decay=self.regularization)
        return optimizer

    def on_train_epoch_end(self):
        # compute metrics
        train_accuracy = self.train_acc.compute()
        train_ba = self.train_ba.compute()
        # log metrics
        self.log("epoch_train_accuracy", train_accuracy)
        self.train_acc.reset()
        self.train_ba.reset()

    def on_validation_epoch_end(self):
        # compute metrics
        val_accuracy = self.val_acc.compute()
        val_ba = self.val_ba.compute()
        # log metrics
        self.log("val_accuracy", val_accuracy, prog_bar=True)
        self.log("val_ba", val_ba, prog_bar=True)
        # reset all metrics
        self.val_acc.reset()
        self.val_ba.reset()

    def on_test_epoch_end(self):
        # compute metrics
        val_accuracy = self.val_acc.compute()
        val_ba = self.val_ba.compute()
        # log metrics
        self.log("test_accuracy", val_accuracy)
        self.log("test_ba", val_ba)
        # reset all metrics
        self.val_acc.reset()
        self.val_ba.reset()
        return val_accuracy, val_ba


class DefaultRegressor(pl.LightningModule):

    def __init__(self, input_shape: int, output_size: int = 1, architecture: 'str' = 'linear', regularization=0.) -> None:
        super(DefaultRegressor, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.architecture = architecture
        self.model = self.__create_model()
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.loss = nn.MSELoss()
        self.regularization = regularization

    def __create_model(self) -> nn.Module:
        if self.architecture == 'linear':
            return networks.LinearModel(np.prod(self.input_shape), self.output_size)
        elif self.architecture == 'three_layer_nn':
            return networks.ThreeLayerNNModel(np.prod(self.input_shape), self.output_size)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        if 'conv' not in self.architecture:
            x = x.view((x.size(0), -1))
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y, sample_weight = batch
        y = y.view(-1, 1)
        y_hat = self.forward(x)
        mae = self.train_mae(y_hat, y)
        self.train_mse.update(y_hat, y)
        loss = self.loss(y_hat, y) # - 1e-4*torch.sum(s * torch.log(s_clip))
        self.log('train_mse', loss, prog_bar=True)
        self.log("train_mae", mae, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.val_mae.update(y_hat, y)
        self.val_mse.update(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.val_mae.update(y_hat, y)
        self.val_mse.update(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, weight_decay=self.regularization)
        return optimizer

    def on_train_epoch_end(self):
        # compute metrics
        train_mse = self.train_mse.compute()
        train_mae = self.train_mae.compute()
        # log metrics
        self.log("epoch_train_mae", train_mae)
        self.log("epoch_train_mse", train_mse)
        self.train_mae.reset()
        self.train_mse.reset()

    def on_validation_epoch_end(self):
        # compute metrics
        val_mae = self.val_mae.compute()
        val_mse = self.val_mse.compute()
        # log metrics
        self.log("val_mae", val_mae, prog_bar=True)
        self.log("val_mse", val_mse, prog_bar=True)
        # reset all metrics
        self.val_mae.reset()
        self.val_mse.reset()

    def on_test_epoch_end(self):
        # compute metrics
        val_mae = self.val_mae.compute()
        val_mse = self.val_mse.compute()
        # log metrics
        self.log("test_mae", val_mae)
        self.log("test_mse", val_mse)
        # reset all metrics
        self.val_mae.reset()
        self.val_mse.reset()
        return val_mse, val_mae


class DefaultSVR(pl.LightningModule):

    def __init__(self, input_shape: int, kernel: np.ndarray, output_size: int = 1, kernel_type: 'str' = 'rbf', regularization=0., mask=None) -> None:
        super(DefaultSVR, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.kernel_type = kernel_type
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.loss = nn.MSELoss()
        self.regularization = regularization
        self.mask = mask
        self.model = self.__create_model(kernel)

    def __create_model(self, kernel) -> nn.Module:
        return networks.SVCRBF(kernel, self.output_size, mask=self.mask)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y, sample_weight = batch
        y = y.view(-1, 1)
        y_hat = self.forward(x)
        mae = self.train_mae(y_hat, y)
        self.train_mse.update(y_hat, y)
        loss = self.loss(y_hat, y) # - 1e-4*torch.sum(s * torch.log(s_clip))
        self.log('train_mse', loss, prog_bar=True)
        self.log("train_mae", mae, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.val_mae.update(y_hat, y)
        self.val_mse.update(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.val_mae.update(y_hat, y)
        self.val_mse.update(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, weight_decay=self.regularization)
        return optimizer

    def on_train_epoch_end(self):
        # compute metrics
        train_mse = self.train_mse.compute()
        train_mae = self.train_mae.compute()
        # log metrics
        self.log("epoch_train_mae", train_mae)
        self.log("epoch_train_mse", train_mse)
        self.train_mae.reset()
        self.train_mse.reset()

    def on_validation_epoch_end(self):
        # compute metrics
        val_mae = self.val_mae.compute()
        val_mse = self.val_mse.compute()
        # log metrics
        self.log("val_mae", val_mae, prog_bar=True)
        self.log("val_mse", val_mse, prog_bar=True)
        # reset all metrics
        self.val_mae.reset()
        self.val_mse.reset()

    def on_test_epoch_end(self):
        # compute metrics
        val_mae = self.val_mae.compute()
        val_mse = self.val_mse.compute()
        # log metrics
        self.log("test_mae", val_mae)
        self.log("test_mse", val_mse)
        # reset all metrics
        self.val_mae.reset()
        self.val_mse.reset()
        return val_mse, val_mae


