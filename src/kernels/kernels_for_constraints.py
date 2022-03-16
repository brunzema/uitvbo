import torch
import gpytorch
from abc import abstractmethod
from typing import List
from utils.torch_utils import _match_batch_dims


class BaseKernelForConvexityConstraints(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def k_x1i_ddx2j(self, x1, x2, j=0):
        # derivative of kernel function ( \frac{\parial}{\partial^2 y_j} K(x,y) )
        raise NotImplementedError

    @abstractmethod
    def k_ddx1i_ddx2j(self, x1, x2, i=0, j=0):
        # derivative of kernel function ( \frac{\parial}{\partial^2 x_i \partial^2 y_j} K(x,y) )
        raise NotImplementedError

    def construct_k_ddx1_ddx2(self, x1, x2, bounded=False):
        x1_, x2_ = self.get_active_dims(x1, x2)

        if bounded:
            K_xv = 0

        if x1_.shape[-2] == x2_.shape[-2]:
            # calculate blocks for upper triangle
            k_ddx1_ddx2_blocks = []
            for i in range(self.nr_constraints):
                row = []
                for l in range(self.nr_constraints - i):
                    j = i + l
                    k_ddx1_ddx2_temp = self.k_ddx1i_ddx2j(x1_, x2_, i=i, j=j)
                    row.append(k_ddx1_ddx2_temp)
                k_ddx1_ddx2_blocks.append(row)

            # Create upper triangle
            k_ddx1_ddx2 = torch.cat(k_ddx1_ddx2_blocks[0], dim=-1)
            n_cols = k_ddx1_ddx2.shape[-1]
            for i in range(self.nr_constraints - 1):
                tmp = torch.cat(k_ddx1_ddx2_blocks[i + 1], dim=-1)
                n_rows = tmp.shape[-2]
                blanks = torch.zeros((n_rows, n_cols - tmp.shape[-1]))
                _, blanks = _match_batch_dims(tmp, blanks)
                nxt_block_row = torch.cat((blanks, tmp), dim=-1)
                k_ddx1_ddx2 = torch.cat((k_ddx1_ddx2, nxt_block_row), dim=-2)

            # fill lower triangle
            k_ddx1_ddx2 = torch.triu(k_ddx1_ddx2)
            diagonal = torch.diagonal(k_ddx1_ddx2, dim1=-2, dim2=-1)
            diagonal = torch.diag_embed(diagonal)
            k_ddx1_ddx2 = k_ddx1_ddx2 + k_ddx1_ddx2.transpose(dim0=-2, dim1=-1) - diagonal

            return k_ddx1_ddx2
        else:
            k_ddx_ddy = torch.zeros(len(x1_) * self.nr_constraints, len(x2_) * self.nr_constraints)
            for i in range(self.nr_constraints):
                for j in range(self.nr_constraints):
                    from_x, to_x = len(x1_) * i, len(x1_) * (i + 1)
                    from_y, to_y = len(x2_) * j, len(x2_) * (j + 1)
                    k_ddx_ddy[from_x:to_x, from_y:to_y] = self.k_ddx1i_ddx2j(x1_, x2_, i=i, j=j)
            return k_ddx_ddy

    def construct_k_x1_ddx2(self, x1, x2):
        x1_, x2_ = self.get_active_dims(x1, x2)

        k_x1_ddx2 = None
        for j in range(self.nr_constraints):
            k_x1_ddx2_tmp = self.k_x1i_ddx2j(x1_, x2_, j=j)
            if k_x1_ddx2 is None:
                k_x1_ddx2 = k_x1_ddx2_tmp
            else:
                k_x1_ddx2 = torch.cat((k_x1_ddx2, k_x1_ddx2_tmp), dim=-1)
        return k_x1_ddx2

    def get_active_dims(self, x1, x2):
        if self.active_dims is not None:
            # Select the active dims
            x1_ = x1.index_select(-1, self.active_dims)
            x2_ = x2.index_select(-1, self.active_dims)

            # Give x1_ and x2_ a last dimension, if necessary
            if x1_.ndimension() == 1:
                x1_ = x1_.unsqueeze(1)
            if x2_.ndimension() == 1:
                x2_ = x2_.unsqueeze(1)
            if not x1_.size(-1) == x2_.size(-1):
                raise RuntimeError("x1_ and x2_ must have the same number of dimensions!")
            return x1_, x2_
        else:
            return x1, x2


class RBFKernelForConvexityConstraints(BaseKernelForConvexityConstraints):
    has_lengthscale = True

    def __init__(self, constrained_dims: List[int], **kwargs):
        super().__init__(**kwargs)
        self.evaluate_derivatives = False
        self.constrained_dims = constrained_dims  # list of dims
        self.nr_constraints = len(self.constrained_dims)

    def forward(self, x1, x2, diag=False, **params):
        x1_ = self.scale(x1)
        x2_ = self.scale(x2)
        k_x1x2 = self.covar_dist(x1_, x2_, square_dist=True, diag=diag,
                                 dist_postprocess_func=self.postprocess_rbf,
                                 postprocess=True,
                                 **params)
        return k_x1x2

    def scale(self, unscaled_input):
        return unscaled_input.div(self.lengthscale)

    def k_x1i_ddx2j(self, x1, x2, j=0):
        # second partial derivative w.r.t. the second input x2
        base = self.forward(x1, x2)
        x1_ = self.scale(x1)
        x2_ = self.scale(x2)

        # get batchsize
        batch_shape = x1_.shape[:-2]

        norm_lengthscale_j = 1 / (self.lengthscale[:, j] ** 2)
        sq_dist_j = self.covar_dist(x1_[..., j].unsqueeze(-1),
                                    x2_[..., j].unsqueeze(-1), square_dist=True)
        out = base * norm_lengthscale_j * (sq_dist_j - 1)
        return out

    def k_ddx1i_ddx2j(self, x1, x2, i=0, j=0):
        # second partial derivative w.r.t. the first input x1 and second input x2
        base = self.forward(x1, x2)
        x1_ = self.scale(x1)
        x2_ = self.scale(x2)

        norm_lengthscale_ij = 1 / (self.lengthscale[:, i] ** 2 * self.lengthscale[:, j] ** 2)
        sq_dist_i = self.covar_dist(x1_[..., i].unsqueeze(-1),
                                    x2_[..., i].unsqueeze(-1), square_dist=True)
        sq_dist_j = self.covar_dist(x1_[..., j].unsqueeze(-1),
                                    x2_[..., j].unsqueeze(-1), square_dist=True)

        if i == j:
            return base * norm_lengthscale_ij * (sq_dist_i * sq_dist_i - 6 * sq_dist_i + 3)
        else:
            return base * norm_lengthscale_ij * (sq_dist_i * sq_dist_j - sq_dist_i - sq_dist_j + 1)

    def postprocess_rbf(self, dist_mat):
        return dist_mat.div_(-2).exp_()
