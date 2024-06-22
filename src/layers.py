import torch
from torch import nn
import numpy as np



class E2EFSMaskBase(nn.Module):

    def __init__(self, input_shape, n_features_to_select,
                 heatmap_momentum=.99999,
                 epsilon=1e-8,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(E2EFSMaskBase, self).__init__()
        self.units = np.prod(input_shape)
        self.epsilon = epsilon
        self.kernel = nn.parameter.Parameter(torch.ones(input_shape, **factory_kwargs))
        self.register_buffer('heatmap', torch.zeros_like(self.kernel, **factory_kwargs))
        self.register_buffer('moving_units', torch.tensor(n_features_to_select, **factory_kwargs))
        self.register_buffer('moving_T', torch.tensor(0., **factory_kwargs))
        self.register_buffer('moving_factor', torch.tensor(0., **factory_kwargs))
        self.heatmap_momentum = heatmap_momentum
        self.n_features_to_select = n_features_to_select

    def forward(self, x):
        output = x * self.kernel_activation()
        return output

    def update_buffers(self):
        self.heatmap = self.heatmap_momentum * self.heatmap + (1. - self.heatmap_momentum) * torch.sign(self.kernel_activation())
        self.moving_T = self.moving_T + 1
        self.moving_factor = torch.where(
            torch.less(self.moving_T, self.warmup_T),
            self.start_alpha,
            (self.start_alpha + (1. - self.start_alpha) * (self.moving_T - self.warmup_T) / self.T).clamp(max=self.alpha_M)
        )

    def get_penalty(self):
        x = self.kernel
        t = x / torch.max(torch.abs(x))
        p = torch.where(torch.less(t, self.epsilon), torch.zeros_like(x), x)
        m = torch.sum(torch.greater(p, 0.))
        sum_x = torch.sum(p)
        moving_units = torch.where(torch.less_equal(m, self.n_features_to_select), m,
                                   (1. - self.moving_decay) * self.moving_units)
        l_units = torch.abs(moving_units - sum_x)

        cost = torch.sum(p * (1. - p)) + 2. * l_units
        return cost

    def kernel_activation(self):
        t = self.kernel / torch.max(torch.abs(self.kernel))
        s = torch.where(torch.less(t, self.epsilon), torch.zeros_like(self.kernel), self.kernel)
        return s

    def kernel_constraint(self):
        self.kernel.data = self.kernel.data.clamp(min=0., max=1.)
        # if self.get_n_alive() <= self.n_features_to_select:
        #     _, pos = torch.topk(self.heatmap, self.units)
        #     self.kernel.data[pos[self.n_features_to_select:]] = 0.
        # if self.get_n_alive() <= self.n_features_to_select:
        #     _, pos = torch.topk(self.heatmap, self.n_features_to_select)
        #     self.kernel.data = torch.zeros_like(self.kernel.data)
        #     self.kernel.data[pos] = 1.

    def force_kernel(self):
        _, pos = torch.topk(self.heatmap, self.n_features_to_select)
        self.kernel.data = torch.zeros_like(self.kernel.data)
        self.kernel.data[pos] = 1.

    def get_n_alive(self):
        return (self.kernel_activation() > 0).sum()


class E2EFSSoftMask(E2EFSMaskBase):

    def __init__(self, input_shape, n_features_to_select,
                 decay_factor=.75,
                 T=20000,
                 warmup_T=2000,
                 start_alpha=.0,
                 alpha_N=.99,
                 epsilon=1e-8,
                 device=None, dtype=None):

        self.decay_factor = decay_factor
        self.T = T
        self.warmup_T = warmup_T
        self.start_alpha = start_alpha
        self.cont_T = 0
        self.alpha_M = alpha_N
        super(E2EFSSoftMask, self).__init__(input_shape=input_shape,
                                        n_features_to_select=n_features_to_select,
                                        heatmap_momentum=(T - 1.) / T,
                                        epsilon=epsilon,
                                        device=device, dtype=dtype)

        self.register_buffer('moving_decay', torch.tensor(decay_factor, device=device, dtype=dtype))

    def update_buffers(self):
        super(E2EFSSoftMask, self).update_buffers()
        self.moving_decay = torch.where(
            torch.less(self.moving_factor, self.alpha_M),
            self.moving_decay,
            (self.moving_decay + self.epsilon).clamp(min=.75)
        )


class E2EFSMask(E2EFSSoftMask):

    def __init__(self, input_shape, n_features_to_select,
                 device=None, dtype=None):

        super(E2EFSMask, self).__init__(input_shape=input_shape,
                                    n_features_to_select=n_features_to_select,
                                    decay_factor=0.,
                                    T=10000, device=device, dtype=dtype)


class E2EFSRamkingMask(E2EFSSoftMask):

    def __init__(self, input_shape, n_features_to_select=1,
                 speedup=4.,
                 device=None, dtype=None):

        super(E2EFSRamkingMask, self).__init__(input_shape=input_shape,
                                    n_features_to_select=n_features_to_select,
                                    decay_factor=0.,
                                    T=20000, device=device, dtype=dtype)
        self.register_buffer('speedup', torch.tensor(speedup, device=device, dtype=dtype))

    def update_buffers(self):
        super(E2EFSSoftMask, self).update_buffers()
        units = torch.tensor(self.units)
        self.moving_units = torch.where(
            torch.less_equal(self.moving_T, self.warmup_T),
            (1. - self.start_alpha) * units,
            (units * torch.pow(
                1. / units,  self.speedup * (self.moving_T - self.warmup_T) / self.T
            )).clamp(min=self.alpha_M)
        )
