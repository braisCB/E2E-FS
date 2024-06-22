import torch
from torch import nn
import numpy as np


class LinearModel(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(LinearModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        return self.model(x)


class ThreeLayerNNModel(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(ThreeLayerNNModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.BatchNorm1d(50),
            nn.SiLU(),
            nn.Linear(50, 25),
            nn.BatchNorm1d(25),
            nn.SiLU(),
            nn.Linear(25, 10),
            nn.BatchNorm1d(10),
            nn.SiLU(),
            nn.Linear(10, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class SVCRBF(nn.Module):

    def __init__(self, kernel, output_dim, gamma=.001, mask=None, device=None, dtype=None):
        super(SVCRBF, self).__init__()
        self.register_buffer('kernel', torch.tensor(kernel))
        self.output_dim = output_dim
        self.gamma = gamma
        self.mask = mask
        self.model = nn.Sequential(
            nn.Linear(self.kernel.size(0), output_dim)
        )

    def forward(self, x):
        x_kernel = self.__apply_kernel(x)
        return self.model(x_kernel)

    def __apply_kernel(self, x):
        kernel = self.kernel.type(x.dtype)
        if self.mask is not None:
            kernel = self.mask(kernel)
        xu = torch.unsqueeze(x, dim=1)
        kernelu = torch.unsqueeze(kernel, dim=0)
        diff = xu - kernelu
        return self.gamma * torch.square(diff).sum(dim=2)
