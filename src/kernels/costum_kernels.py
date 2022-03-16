import gpytorch
import torch
from gpytorch.constraints import Interval
import time


class TemporalKernelB2P(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, epsilon=0.08, **kwargs):  # 0.01
        super().__init__(**kwargs)

        self.epsilon = epsilon

    # this is the kernel function
    def forward(self, x1, x2, **params):
        base = 1 - self.epsilon
        # calculate the distance between inputs
        exp = torch.abs(self.covar_dist(x1, x2, square_dist=False)) / 2
        out = torch.pow(base, exp)
        return out


class WienerKernel(gpytorch.kernels.Kernel):  # vorlesung von Phillip Henning
    is_stationary = False

    def __init__(self, c0, sigma_hat_squared=0.5, out_max=2, **kwargs):
        super().__init__(**kwargs)

        self.max_var = out_max
        self.sigma_hat_squared = sigma_hat_squared
        self.c0 = c0

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # d will always be 1, as it is the time dimenaion! Therefore we can squeeze the inputs
        if x1.ndim == 2:  # 'normal' mode
            x1, x2 = x1.squeeze(x1.ndim - 1), x2.squeeze(x2.ndim - 1)
            meshed_x1, meshed_x2 = torch.meshgrid(x1, x2)
            return self.evaluate_kernel(meshed_x1, meshed_x2)

        else:  # 'batch' mode
            # old
            # x1squeezed, x2squeezed = x1.squeeze(x1.ndim - 1), x2.squeeze(x2.ndim - 1)
            # t0 = time.time()
            # out = torch.empty((1, x1squeezed.shape[1], x2squeezed.shape[1]))
            # for batch in range(x1squeezed.shape[0]):
            #     x1_batch = x1squeezed[batch, :]
            #     x2_batch = x2squeezed[batch, :]
            #
            #     meshed_x1, meshed_x2 = torch.meshgrid(x1_batch, x2_batch)
            #     new_out = self.evaluate_kernel(meshed_x1, meshed_x2).unsqueeze(0)
            #
            #     out = torch.cat((out, new_out), dim=0)
            # out1 = out[1:, :, :]
            # print('Loop:', time.time() - t0)

            # t0 = time.time()
            meshed_x1 = torch.tile(x1, (1, 1, x2.shape[1]))
            meshed_x2 = torch.tile(x2.transpose(dim0=-2, dim1=-1), (1, x1.shape[1], 1))
            out = self.evaluate_kernel(meshed_x1, meshed_x2)
            return out

    def evaluate_kernel(self, meshed_x1, meshed_x2):
        step = torch.min(meshed_x1, meshed_x2) - self.c0
        out = step * self.sigma_hat_squared
        return out


class ConstantKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, constant=1, **kwargs):
        super().__init__(**kwargs)
        self.constant = constant

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        return torch.ones_like(self.covar_dist(x1, x2))


# defined only for t>=0
class GeometricWienerKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, t, sigma=0.5, **kwargs):
        super().__init__(**kwargs)

        self.sigma = sigma
        self.c0 = t

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # d will always be 1, as it is the time dimenaion! Therefore we can squeeze it
        x1, x2 = x1.squeeze(x1.ndim - 1), x2.squeeze(x2.ndim - 1)

        if x1.ndim == 1:  # 'normal' mode
            meshed_x1, meshed_x2 = torch.meshgrid(x1, x2)
            return self.evaluate_kernel(meshed_x1, meshed_x2)
        else:  # batch mode

            out = torch.empty((1, x1.shape[1], x2.shape[1]))
            for batch in range(x1.shape[0]):
                x1_batch = x1[batch, :]
                x2_batch = x2[batch, :]

                meshed_x1, meshed_x2 = torch.meshgrid(x1_batch, x2_batch)
                new_out = self.evaluate_kernel(meshed_x1, meshed_x2).unsqueeze(0)

                out = torch.cat((out, new_out), dim=0)
            return out[1:, :, :]

    def evaluate_kernel(self, meshed_x1, meshed_x2):
        step = torch.min(meshed_x1, meshed_x2) - self.c0
        out = step * self.sigma ** 2
        return out


class TemporalKernelUI(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, epsilon_prior=0.08, **kwargs):  # 0.01
        super().__init__(**kwargs)

        self.epsilon = epsilon_prior

    # this is the kernel function
    def forward(self, x1, x2, **params):
        base = 1 - self.epsilon
        # calculate the distance between inputs
        exponent = torch.abs(self.covar_dist(x1, x2, square_dist=False)) / -2.
        out = torch.exp(exponent)
        return out
