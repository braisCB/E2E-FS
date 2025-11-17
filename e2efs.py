import numpy as np
import torch
from src import default_models, layers
from src.e2efs_models import E2EFSModel
from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
from copy import deepcopy
from src.dataloaders import FastTensorDataLoader
from src.callbacks import MyEarlyStopping


class E2EFSBase:

    def __init__(self, n_features_to_select, network=None, mask_name='E2EFSSoftMask', precision='32',
                 balanced=True, regularization='default'):
        self.n_features_to_select = n_features_to_select
        self.network = network
        self.model = None
        self.mask_name = mask_name
        self.precision = precision
        self.balanced = balanced
        self.task = None
        self.regularization = regularization

    def __build_mask__(self, input_shape):
        return getattr(layers, self.mask_name)(input_shape=input_shape, n_features_to_select=self.n_features_to_select)

    def __build_model__(self, X, y, mask=None):
        input_shape = X.shape[1:]
        mask_shape = input_shape if mask is None else mask.shape
        e2efs_mask = self.__build_mask__(input_shape=mask_shape)
        if mask is not None:
            e2efs_mask.kernel.data = mask
        if self.task == 'classification':
            output_size = len(np.unique(y))
        else:
            output_size = y.shape[-1] if len(y.shape) > 1 else 1
        if isinstance(self.network, pl.LightningModule):
            network_model = deepcopy(self.network)
        elif self.network == 'svr':
            network_model = default_models.DefaultSVR(input_shape=input_shape, kernel=X, output_size=output_size,
                                                      kernel_type='rbf', regularization=0.1, mask=e2efs_mask)
        else:
            if self.network is None:
                architecture = self.__select_default_architecture(X, y)
            else:
                architecture = self.network
            reg = self.__select_default_regularization(architecture, X) if self.regularization == 'default' else self.regularization
            default_model = default_models.DefaultRegressor if self.task == 'regression' else default_models.DefaultClassifier
            network_model = default_model(input_shape=input_shape, output_size=output_size, architecture=architecture, regularization=reg)
        return E2EFSModel(network=network_model, e2efs_layer=e2efs_mask)

    def __select_default_architecture(self, X, y):
        nsamples, nfeats = len(X), np.prod(X.shape[1:])
        ratio = nsamples / nfeats
        if ratio < 1.:
            print('Selecting linear model with weight_decay =', 100. / len(X))
            return 'linear'
        else:
            print('Selecting three_layer_nn model with weight_decay =', 1e-3)
            return 'three_layer_nn'

    def __select_default_task(self, X, y):
        if np.issubdtype(y.dtype, np.integer):
            print('Switching default task to classification')
            return 'classification'
        else:
            print('Switching default task to regression')
            return 'regression'

    def __select_default_regularization(self, architecture, X):
        if architecture == 'linear':
            return 10. / len(X)
        else:
            return 1e-3

    def __create_dataloader(self, X, y, batch_size=32, shuffle=False):
        device = self.model.e2efs_layer.kernel.device
        dtype = self.model.e2efs_layer.kernel.dtype
        y_tensor = torch.LongTensor(y, device=device) if self.task == 'classification' else torch.tensor(y, device=device, dtype=dtype)
        if shuffle and self.balanced and self.task == 'classification':
            num_classes = len(np.unique(y))
            y_onehot = torch.nn.functional.one_hot(y_tensor, num_classes)
            class_weight = y_onehot.size(0) / torch.sum(y_onehot, dim=0)
            class_weight = num_classes * class_weight / class_weight.sum()
            sample_weights = class_weight[y_tensor]
        else:
            sample_weights = torch.ones(y_tensor.size(), device=device, dtype=dtype)
        if shuffle:
            # dataset = TensorDataset(torch.tensor(X, device=device, dtype=dtype), y_tensor, sample_weights)
            dataloader = FastTensorDataLoader(
                torch.tensor(X, device=device, dtype=dtype), y_tensor, sample_weights,
                batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
            )
        else:
            # dataset = TensorDataset(torch.tensor(X, device=device, dtype=dtype), y_tensor)
            dataloader = FastTensorDataLoader(
                torch.tensor(X, device=device, dtype=dtype), y_tensor,
                batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
            )
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=0)
        return dataloader

    def __fit_model(self, X, y, validation_data=None, batch_size=32, trainer_opts=dict(), verbose=True):
        train_loader = self.__create_dataloader(X, y, shuffle=True, batch_size=batch_size)
        val_loader = None
        if validation_data is not None:
            val_X, val_y = validation_data
            val_loader = self.__create_dataloader(val_X, val_y, shuffle=False, batch_size=batch_size)
        trainer = pl.Trainer(barebones=True, **trainer_opts)
        # trainer = MyTrainer(enable_model_summary=verbose, enable_progress_bar=False, profiler='simple', **trainer_opts)
        trainer.fit(self.model, train_loader, val_loader)
        self.model.fitted = True
        self.model.e2efs_layer.force_kernel()
        return self

    def fit(self, X, y, mask=None, validation_data=None, batch_size=32, max_epochs=500, verbose=True):
        self.task = self.__select_default_task(X, y)
        self.model = self.__build_model__(X, y, mask)
        trainer_opts = {
            'callbacks': [
                MyEarlyStopping(
                    monitor="nfeats", mode="min", min_delta=0, stopping_threshold=self.n_features_to_select + 1,
                    patience=1000
                )
            ],
            'enable_checkpointing': False,
            'precision': self.precision,
            'max_epochs': max_epochs
        }
        return self.__fit_model(X, y, validation_data, batch_size, trainer_opts=trainer_opts, verbose=verbose)

    def fine_tune(self, X, y, validation_data=None, batch_size=32, max_epochs=1, verbose=True):
        if self.model is None or not self.model.fitted:
            raise Exception('Model not training, Did you call fit() before?')
        trainer_opts = {'max_epochs': max_epochs, 'enable_checkpointing': False, 'precision': self.precision}
        return self.__fit_model(X, y, validation_data, batch_size, trainer_opts=trainer_opts, verbose=verbose)

    def evaluate(self, X, y, batch_size=32, verbose=True):
        if self.model is None or not self.model.fitted:
            raise Exception('Model not training, Did you call fit() before?')
        trainer_opts = {'enable_checkpointing': False, 'precision': self.precision}
        test_loader = self.__create_dataloader(X, y, shuffle=False, batch_size=batch_size)
        trainer = pl.Trainer(enable_model_summary=verbose, enable_progress_bar=verbose, **trainer_opts)
        trainer.test(self.model, test_loader)
        metrics = {k: v.detach().cpu().item() for k, v in trainer.logged_metrics.items()}
        return metrics

    def predict(self, X, batch_size=32, verbose=True):
        if self.model is None or not self.model.fitted:
            raise Exception('Model not training, Did you call fit() before?')
        device = self.model.e2efs_layer.kernel.device
        dtype = self.model.e2efs_layer.kernel.dtype
        dataset = TensorDataset(torch.tensor(X, device=device, dtype=dtype))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        trainer_opts = {'enable_checkpointing': False, 'precision': self.precision}
        trainer = pl.Trainer(enable_model_summary=verbose, enable_progress_bar=verbose, **trainer_opts)
        output = torch.cat(trainer.predict(self.model, dataloader), dim=0).cpu().numpy()
        return output

    def get_ranking(self):
        _, rank = torch.topk(self.model.e2efs_layer.heatmap, self.model.e2efs_layer.units)
        return rank.cpu().numpy()

    def get_mask(self):
        return self.model.e2efs_layer.kernel_activation().detach().cpu().numpy()


class E2EFSSoft(E2EFSBase):

    def __init__(self, n_features_to_select, **kwargs):
        super(E2EFSSoft, self).__init__(n_features_to_select, **kwargs)


class E2EFS(E2EFSBase):

    def __init__(self, n_features_to_select, **kwargs):
        super(E2EFS, self).__init__(n_features_to_select, **kwargs)


class E2EFSRanking(E2EFSBase):

    def __init__(self, **kwargs):
        super(E2EFSRanking, self).__init__(1, **kwargs)

