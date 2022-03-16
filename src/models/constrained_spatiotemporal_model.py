import time

import gpytorch
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from src.kernels.kernels_for_constraints import RBFKernelForConvexityConstraints
from src.kernels.costum_kernels import WienerKernel, TemporalKernelB2P, ConstantKernel
from src.samplers.minimax_tilting_sampler import TruncatedMVN
from utils.torch_utils import _match_batch_dims, _match_dtype
import torch

from typing import Any, Union, List
from torch import Tensor
from gpytorch import settings
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.broadcasting import _mul_broadcast_shape
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn


class ConstrainedGPThroughTime(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self,
                 train_x: Tensor,
                 train_y: Tensor,
                 constrained_dims: List[int],
                 type_of_forgetting: str = 'UI',
                 lengthscale_constraint=None,
                 lengthscale_hyperprior=None,
                 noise_constraint=None,
                 outputscale_constraint_spatio=None,
                 outputscale_hyperprior_spatio=None,
                 outputscale_constraint_temporal=None,
                 outputscale_hyperprior_temporal=None,
                 prev_trunc_samples=None,
                 prior_mean=0,
                 forgetting_factor=0.2, ):

        # check dimensions
        if train_y is not None:
            train_y = train_y.squeeze(-1)

        # constrain homoskedastic noise
        if noise_constraint is None:
            noise_constraint = gpytorch.constraints.GreaterThan(1e-6)

        # init likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint)

        # initialize model
        super(ConstrainedGPThroughTime, self).__init__(train_x, train_y, likelihood)

        # track constrained dimensions
        self.constr_dims = constrained_dims
        self.spatio_dims = train_x.shape[-1] - 1
        if train_x.shape[-1] == len(constrained_dims):
            print('Temporal dimensions can not be convexified!')

        # split kernel into spatiotemporal kernel K = K_s * K_t
        self.spatio_kernel = gpytorch.kernels.ScaleKernel(
            RBFKernelForConvexityConstraints(constrained_dims=constrained_dims,
                                             ard_num_dims=self.spatio_dims,
                                             active_dims=range(self.spatio_dims),
                                             lengthscale_prior=lengthscale_hyperprior,
                                             lengthscale_constraint=lengthscale_constraint),
            outpurtscale_prior=outputscale_hyperprior_spatio,
            outputscale_constraint=outputscale_constraint_spatio)

        # Initialize lengthscale and outputscale to mean of priors.
        if lengthscale_hyperprior is not None:
            self.spatio_kernel.base_kernel.lengthscale = lengthscale_hyperprior.mean
        if outputscale_hyperprior_spatio is not None:
            self.spatio_kernel.outputscale = outputscale_hyperprior_spatio.mean
        else:
            self.spatio_kernel.outputscale = 1
        self.spatio_kernel.outputscale = 1
        self.spatio_kernel.raw_outputscale.requires_grad = False

        # specify forgetting strategy
        if type_of_forgetting == 'UI':  # wiener prozess kernel
            self.forgetting_factor = forgetting_factor
            sigma_w_squared = self.forgetting_factor / self.spatio_kernel.outputscale.clone().detach()
            c0 = - 1 / sigma_w_squared
            self.temporal_kernel = WienerKernel(c0=c0,  # start at t= -10 to have higher flexibility in the mean
                                                sigma_hat_squared=sigma_w_squared,
                                                active_dims=self.spatio_dims)
            self.spatio_kernel.raw_outputscale.requires_grad = False

        elif type_of_forgetting == 'B2P_SE':  # squared-exponential kernel
            self.temporal_kernel = gpytorch.kernels.ScaleKernel(
                RBFKernel(active_dims=self.spatio_dims),
                outpurtscale_prior=outputscale_hyperprior_temporal,
                outputscale_constraint=outputscale_constraint_temporal)
            self.temporal_kernel.base_kernel.lengthscale = 52.
            self.temporal_kernel.outputscale = 5

        elif type_of_forgetting == 'B2P_OU':  # ohrnstrein-uhlenbeck kernel
            self.temporal_kernel = TemporalKernelB2P(epsilon=forgetting_factor,
                                                     active_dims=self.spatio_dims)
            self.spatio_kernel.outputscale = 1
            self.spatio_kernel.raw_outputscale.requires_grad = False
        elif type_of_forgetting is None:
            self.temporal_kernel = ConstantKernel(active_dims=self.spatio_dims)
        else:
            raise NotImplementedError('Only Uncertainty-Injection (“UI“) and '
                                      'Back-2-Prior forgetting ("B2P_SE", "B2P_OU") are implemented.')

        self.type_of_forgetting = type_of_forgetting
        self.covar_module = self.spatio_kernel * self.temporal_kernel

        self.mean_module = ConstantMean()
        if prior_mean != 0:
            self.mean_module.initialize(constant=prior_mean)
        self.mean_module.constant.requires_grad = False

        # factors, that do not need to be recomputed
        self.C = None
        self.U = None
        self.v1 = None
        self.A1 = None
        self.B1 = None
        self.L1 = None
        self.L = None
        self.covar_train_train_noisy = None
        self.covar_train_ddxv = None
        self.covar_ddxv_ddxv = None
        self.centered_mean = None
        self.centered_mean_nobatch = None
        self.proposal_set = None
        self.optimized_VOP = False

        self.location_noise = 1e-6  # numerical stability -> lowering the probability of hard constraints
        self.bounds = [0, np.inf]  # numpy since sampling algorithm needs numpy bounds
        self.prev_trunc_samples = prev_trunc_samples
        self.prev_post_samples = None
        self.posterior_distribution = None

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def normalize_and_update_forgetting_factor(self):
        if self.type_of_forgetting == 'UI':
            sigma_w = self.forgetting_factor / self.spatio_kernel.outputscale
            c0 = -1 / sigma_w
            self.temporal_kernel.sigma = sigma_w
            self.temporal_kernel.c0 = c0

    def posterior(self, X: Tensor, observation_noise: Union[bool, Tensor] = False, **kwargs: Any
                  ) -> GPyTorchPosterior:

        if 'evaluate_constrained' in kwargs:
            if kwargs['evaluate_constrained'] is True:
                # get obseration points
                self.proposed_VOP = kwargs['virtual_oberservation_points']
                if not isinstance(self.proposed_VOP, Tensor):
                    raise RuntimeError("Virtual observations must be a tensor!")

                n_samples = kwargs['nr_samples']

                # check if batches are used
                batch_mode = len(X.shape) == 3

                # get joint distribution of training and testing 
                full_mean, full_covar = self.get_joint_distribution(X)

                # assign mean and covariances
                train_mean = full_mean[..., :self.num_train]
                test_mean = full_mean[..., self.num_train:]

                covar_train_test = full_covar[..., :self.num_train, self.num_train:].evaluate()
                covar_test_test = full_covar[..., self.num_train:, self.num_train:].evaluate()

                # calculate training covar (can reuse)
                measurement_noise = self.likelihood.noise_covar.noise
                if not isinstance(self.covar_train_train_noisy, Tensor):
                    covar_train_train = full_covar[..., :self.num_train, :self.num_train].evaluate()
                    covar_XX_noisy = torch.add(covar_train_train, torch.eye(self.num_train) * measurement_noise)
                    self.covar_train_train_noisy = covar_XX_noisy[0, ...] if batch_mode else covar_XX_noisy

                # match dimensions
                X, training_inputs, training_targets = _match_batch_dims(X, self.train_inputs[0], self.train_targets)

                self.centered_mean = training_targets - train_mean
                self.centered_mean_nobatch = self.centered_mean[0, ...] if batch_mode else self.centered_mean
                # optimizing locations is independent on test data, therefore no batches needed for calculation

                # optimize virtual locations
                if not isinstance(self.optimized_VOP, Tensor):
                    if 'optimize_VOL' in kwargs:
                        if kwargs['optimize_VOL']:
                            # virtual observation locations (VOL)
                            VOL_options = kwargs['VOL_options']
                            self.optimized_VOP, _ = self.find_optimal_VOL(VOL_options)
                        else:
                            self.optimized_VOP = self.proposed_VOP
                    else:
                        self.optimized_VOP = self.proposed_VOP

                # calculate and construct the derivative covar matrices
                covar_train_ddxv, covar_ddxv_ddxv, covar_test_ddxv = self.get_derivative_covars(X, training_inputs)

                # calculate factors (Agrell 2019)
                L, v1, A1, B1, L1 = self.get_factors()
                X, L, v1, A1, B1, L1 = _match_batch_dims(X, L, v1, A1, B1, L1)  # match batch dimensions
                L_transpose = L.transpose(dim0=-2, dim1=-1)

                v2 = torch.triangular_solve(covar_train_test, L, upper=False).solution
                A2 = torch.triangular_solve(v2, L_transpose, upper=True).solution.transpose(dim0=-2, dim1=-1)

                B2 = covar_test_test - v2.transpose(dim0=-2, dim1=-1) @ v2
                B3 = covar_test_ddxv - v2.transpose(dim0=-2, dim1=-1) @ v1
                L1_transpose = L1.transpose(dim0=-2, dim1=-1)  # for numerical stability

                v3 = torch.triangular_solve(B3.transpose(dim0=-2, dim1=-1), L1, upper=False).solution

                A = torch.triangular_solve(v3, L1_transpose, upper=True).solution.transpose(dim0=-2, dim1=-1)
                B = A2 - A @ A1

                # mu_s and and cov_s only needed for sampling - samples can be used for the whole batch since
                # observations and training do not change
                if batch_mode:
                    mu_s = A1[0, ...] @ self.centered_mean_nobatch
                    cov_s = B1[0, ...]
                    self.centered_mean = self.centered_mean.unsqueeze(-1)
                    test_mean = test_mean.unsqueeze(-1)
                else:
                    mu_s = A1 @ self.centered_mean
                    cov_s = B1

                sigma = B2 - v3.transpose(dim0=-2, dim1=-1) @ v3
                Q = psd_safe_cholesky(sigma, max_tries=6)  # for numerical stability

                # reuse samples!
                if self.C is None:
                    C = self.sample_from_truncated_norm(mu_s, cov_s, n_samples=n_samples,
                                                        sampling_method='minimax_tilting')
                    self.C = C
                    self.prev_trunc_samples = C[:, -100:]
                else:
                    C = self.C

                # reshape to batches if needed
                if batch_mode:
                    _, C = _match_batch_dims(X, C)  # reuse samples for whole batch

                # draw samples (dim(x_test) x n_samples) from multivariate standard distribution
                m = torch.distributions.MultivariateNormal(torch.zeros(Q.shape[-1]), torch.eye(Q.shape[-1]))
                # torch.manual_seed(356997) # seed showed to be not beneficial as it was influencing the result
                U = m.rsample(sample_shape=torch.Size([C.shape[-1]])).T

                # calculate mean
                prior_test_mean = (test_mean + B @ self.centered_mean)
                if batch_mode:
                    prior_test_mean = torch.tile(prior_test_mean, (1, 1, C.shape[-1]))
                else:
                    prior_test_mean = torch.tile(prior_test_mean.unsqueeze(-1), (1, C.shape[-1]))

                # convert samples to type of the input
                C, U = _match_dtype(X, C, U)

                # approximate posterior f*| X, Y, C
                fs_sim = prior_test_mean + A @ C + Q @ U
                if 'return_samples' in kwargs:
                    if kwargs['return_samples']:
                        return fs_sim

                post_mean = fs_sim.mean(axis=-1)
                post_cov = fs_sim.var(axis=-1)

                # store random samples from the posterior
                if not batch_mode:
                    self.prev_post_samples = fs_sim

                # convert to diagonal matrix (b x len(test) x len(test))
                diagmat_post_cov = torch.diag_embed(post_cov, offset=0, dim1=-2, dim2=-1)

                # catch nans
                diagmat_post_cov = torch.nan_to_num(diagmat_post_cov, nan=1.0)

                # convert to required gpytorch.distribution.MultivariateNormal
                mvn = MultivariateNormal(post_mean, diagmat_post_cov)
                self.posterior_distribution = GPyTorchPosterior(mvn)
                return self.posterior_distribution

        # return default posterior if not constrained evaluation in kwargs
        self.posterior_distribution = super().posterior(X=X, observation_noise=observation_noise, **kwargs)
        return self.posterior_distribution

    def get_factors(self):
        if not isinstance(self.L, Tensor):
            self.L = psd_safe_cholesky(self.covar_train_train_noisy, max_tries=5)
        L = self.L

        if not isinstance(self.v1, Tensor):
            self.v1 = torch.triangular_solve(self.covar_train_ddxv, self.L, upper=False).solution
        v1 = self.v1

        if not isinstance(self.A1, Tensor):
            self.A1 = torch.triangular_solve(self.v1, self.L.T, upper=True).solution.T
        A1 = self.A1

        if not isinstance(self.B1, Tensor):
            covar_ddxv_ddxv_noisy = self.covar_ddxv_ddxv + torch.eye(
                self.covar_ddxv_ddxv.shape[-1]) * self.location_noise
            self.B1 = covar_ddxv_ddxv_noisy - v1.T @ v1
        B1 = self.B1

        if not isinstance(self.L1, Tensor):
            self.L1 = psd_safe_cholesky(B1, max_tries=5)
        L1 = self.L1

        return L, v1, A1, B1, L1

    def get_derivative_covars(self, X, training_inputs):
        batch_mode = len(X.shape) == 3
        X, virtual_obs_points = _match_batch_dims(X, self.optimized_VOP)

        # get covariances matrices of the partial derivatives of constrained spatio dimensions
        spatio_scale = self.spatio_kernel.outputscale

        training_inputs_nobatch = training_inputs[0, ...] if batch_mode else training_inputs
        virtual_obs_points_nobatch = virtual_obs_points[0, ...] if batch_mode else virtual_obs_points

        # covar(X, X_v)
        if not isinstance(self.covar_train_ddxv, Tensor):
            covar_train_ddxv = self.spatio_kernel.base_kernel.construct_k_x1_ddx2(
                training_inputs_nobatch, virtual_obs_points_nobatch)
            tmpr_covar_train_ddxv = self.temporal_kernel(training_inputs_nobatch,
                                                         virtual_obs_points_nobatch).evaluate()
            tmpr_covar_train_ddxv = torch.tile(tmpr_covar_train_ddxv, (1, len(self.constr_dims)))
            covar_train_ddxv = covar_train_ddxv * tmpr_covar_train_ddxv
            self.covar_train_ddxv = covar_train_ddxv * spatio_scale
        else:
            covar_train_ddxv = self.covar_train_ddxv
        _, covar_train_ddxv = _match_batch_dims(X, covar_train_ddxv)

        # covar(X_v, X_v)
        if not isinstance(self.covar_ddxv_ddxv, Tensor):
            covar_ddxv_ddxv = self.spatio_kernel.base_kernel.construct_k_ddx1_ddx2(
                virtual_obs_points_nobatch, virtual_obs_points_nobatch)
            tmpr_covar_ddxv_ddxv = self.temporal_kernel(virtual_obs_points_nobatch,
                                                        virtual_obs_points_nobatch).evaluate()
            tmpr_covar_ddxv_ddxv = torch.tile(tmpr_covar_ddxv_ddxv,
                                              (len(self.constr_dims), len(self.constr_dims)))
            covar_ddxv_ddxv = covar_ddxv_ddxv * tmpr_covar_ddxv_ddxv
            self.covar_ddxv_ddxv = covar_ddxv_ddxv * spatio_scale
        else:
            covar_ddxv_ddxv = self.covar_ddxv_ddxv
        _, covar_ddxv_ddxv = _match_batch_dims(X, covar_ddxv_ddxv)

        # covar(X_*, X_v)
        covar_test_ddxv = self.spatio_kernel.base_kernel.construct_k_x1_ddx2(X, virtual_obs_points)
        tmpr_covar_test_ddxv = self.temporal_kernel(X, virtual_obs_points).evaluate()
        if batch_mode:
            tmpr_covar_test_ddxv = torch.tile(tmpr_covar_test_ddxv, (1, 1, len(self.constr_dims)))
        else:
            tmpr_covar_test_ddxv = torch.tile(tmpr_covar_test_ddxv, (1, len(self.constr_dims)))
        covar_test_ddxv = covar_test_ddxv * tmpr_covar_test_ddxv
        covar_test_ddxv = covar_test_ddxv * spatio_scale

        return covar_train_ddxv, covar_ddxv_ddxv, covar_test_ddxv

    def get_samples_from_prior(self, X, samples, **kwargs):

        # get joint distribution of training and testing
        full_mean, full_covar = self.get_joint_distribution(X)

        # assign mean and covariances
        test_mean = full_mean[..., self.num_train:]
        covar_test_test = full_covar[..., self.num_train:, self.num_train:].evaluate()

        if 'evaluate_constrained' not in kwargs:
            mvn = MultivariateNormal(test_mean, covar_test_test)
            return GPyTorchPosterior(mvn).sample(sample_shape=torch.Size([samples])).detach().numpy()

        else:
            if kwargs['evaluate_constrained'] is True:
                # get obseration points
                self.proposed_VOP = kwargs['virtual_oberservation_points']
                if not isinstance(self.proposed_VOP, Tensor):
                    raise RuntimeError("Virtual observations must be a tensor!")

                if 'nr_samples' in kwargs:
                    print('Nr. of samples is already defined in the function call.')

                # optimize virtual locations
                if not isinstance(self.optimized_VOP, Tensor):
                    if 'optimize_VOL' in kwargs:
                        if kwargs['optimize_VOL']:
                            # virtual observation locations (VOL)
                            VOL_options = kwargs['VOL_options']
                            self.optimized_VOP, _ = self.find_optimal_VOL(VOL_options)
                        else:
                            self.optimized_VOP = self.proposed_VOP
                    else:
                        self.optimized_VOP = self.proposed_VOP

                _, covar_ddxv_ddxv, covar_test_ddxv = self.get_derivative_covars(X, self.optimized_VOP)

                # calculate factors (Agrell 2019), TODO: Speed up by saving v1, L1, B1
                covar_ddxv_ddxv_noisy = torch.add(covar_ddxv_ddxv,
                                                  torch.eye(covar_ddxv_ddxv.shape[-1]) * self.location_noise)

                B1 = covar_ddxv_ddxv_noisy
                B2 = covar_test_test
                B3 = covar_test_ddxv
                L1 = psd_safe_cholesky(B1)
                L1_transpose = L1.transpose(dim0=-2, dim1=-1)  # for numerical stability

                v3 = torch.triangular_solve(B3.transpose(dim0=-2, dim1=-1), L1, upper=False).solution

                A = torch.triangular_solve(v3, L1_transpose, upper=True).solution.transpose(dim0=-2, dim1=-1)

                sigma = B2 - v3.transpose(dim0=-2, dim1=-1) @ v3
                Q = psd_safe_cholesky(sigma, max_tries=6)

                cov_s = B1
                mu_s = torch.zeros(cov_s.shape[0])

                C = self.sample_from_truncated_norm(mu_s, cov_s, n_samples=samples,
                                                    sampling_method='minimax_tilting')

                # draw samples (dim(x_test) x n_samples) from multivariate standard distribution
                m = torch.distributions.MultivariateNormal(torch.zeros(Q.shape[-1]), torch.eye(Q.shape[-1]))
                # torch.manual_seed(356997) # seed showed to be not beneficial as it was influencing the result
                U = m.rsample(sample_shape=torch.Size([C.shape[-1]])).T
                # calculate mean
                prior_test_mean = torch.tile(test_mean.unsqueeze(-1), (1, C.shape[-1]))

                # convert samples to type of the input
                C, U = _match_dtype(X, C, U)

                # approximate prior f*| C
                fs_sim = prior_test_mean + A @ C + Q @ U
                return fs_sim
            else:
                mvn = MultivariateNormal(test_mean, covar_test_test)
                return GPyTorchPosterior(mvn).sample(sample_shape=torch.Size([samples])).detach()

    def get_samples_from_last_posterior(self, n_samples):
        if isinstance(self.prev_post_samples, Tensor):
            perm = torch.randperm(self.prev_post_samples.size(1))
            idx = perm[:n_samples]
            return self.prev_post_samples[:, idx].clone().detach()
        else:
            sample = self.posterior_distribution.sample(sample_shape=torch.Size([n_samples])).detach().T
            return sample.squeeze(0)

    def reset_samples(self):
        self.C = None
        self.optimized_VOP = None

    def reset_factors(self):
        self.L = None
        self.covar_train_ddxv = None
        self.covar_ddxv_ddxv = None
        self.centered_mean = None
        self.centered_mean_nobatch = None
        self.L1 = None
        self.B1 = None
        self.v1 = None
        self.A1 = None

    def find_optimal_VOL(self, VOL_options):

        # assign options
        p_target = VOL_options['p_target']
        self.proposal_set = VOL_options['proposal_set']
        max_locations = VOL_options['max_locations']
        use_approximation = VOL_options['use_approximation']

        t0 = time.time()
        ov_time_sampled = 0
        ov_time_calc = 0
        ov_time_prob_calc = 0
        nu = max(self.location_noise * norm.ppf(p_target), 0)

        locations = []
        pc_mins = []
        pc_min = 0

        for j in range(1, max_locations + 1):
            # Run optimization on proposal points
            pc_min, x_min, time_sampled, time_calc, time_prob_calc = self._argmin_pc_finite(nu, locations,
                                                                                            use_approximation)
            # update times
            ov_time_sampled += time_sampled
            ov_time_calc += time_calc
            ov_time_prob_calc += time_prob_calc

            # check if selected point satisfies the p_target, else append new location
            if pc_min >= p_target:
                print(f'DONE - Found {j} points. Min. constraint prob = {pc_min:10.3f}. '
                      f'\nTotal time spent = {time.time() - t0:10.3f}s. '
                      f'Sampling time = {ov_time_sampled:10.3f}s. Calculation time = {ov_time_calc:10.3f}s. '
                      f'Calculcate probability time = {ov_time_prob_calc:10.3f}s.')
                return torch.stack(locations), torch.tensor(pc_mins)
            else:
                pc_mins.append(pc_min)
                locations.append(x_min)
                print(f'Currently {j} of max. {max_locations} locations. Current min_prob is {pc_min:.4f}.')

        print(f'DONE - Found {max_locations} points. Min. constraint prob still = {pc_min:10.3f} < {p_target}. '
              f'More constraint points could be nessassary. '
              f'\nTotal time spent = {time.time() - t0:10.3f}s. '
              f'Sampling time = {ov_time_sampled:10.3f}s. Calculation time = {ov_time_calc:10.3f}s. '
              f'Calculcate probability time = {ov_time_prob_calc:10.3f}s.')
        return torch.stack(locations), torch.tensor(pc_mins)

    def _argmin_pc_finite(self, nu, locations, use_approximation):

        constraint_violation_prob = []
        ov_time_sampled = 0
        ov_time_calculated = 0
        ov_time_prob_calculated = 0
        for opt_point in self.proposal_set:
            opt_prob, time_sampled, time_calculated, time_prob_calculated = self._compute_satisfying_constrain_prob(
                opt_point.unsqueeze(0), locations, nu, use_approximation=use_approximation)
            constraint_violation_prob.append(opt_prob)
            ov_time_sampled += time_sampled
            ov_time_calculated += time_calculated
            ov_time_prob_calculated += time_prob_calculated

        # Choose point with lowest probability of satisfying the constraint
        pc_min = min(constraint_violation_prob)
        i_min = constraint_violation_prob.index(pc_min)
        x_min = self.proposal_set[i_min]
        self.proposal_set = np.delete(self.proposal_set, i_min, 0)
        return pc_min, x_min, ov_time_sampled, ov_time_calculated, ov_time_prob_calculated

    def _compute_satisfying_constrain_prob(self, opt_point, chosen_locations, nu, n_samples=1000,
                                           use_approximation=False):
        time_calculated = time.time()
        spatio_scale = self.spatio_kernel.outputscale
        # for initial location
        if not chosen_locations:

            # comupute factors (AGRELL)
            spatio_k_X_ddx = spatio_scale * self.spatio_kernel.base_kernel.construct_k_x1_ddx2(
                self.train_inputs[0],
                opt_point)
            spatio_k_ddx_ddx = spatio_scale * self.spatio_kernel.base_kernel.construct_k_ddx1_ddx2(
                opt_point,
                opt_point)

            # temporal kernel
            tmpr_k_X_ddx = self.temporal_kernel(self.train_inputs[0], opt_point).evaluate()
            tmpr_k_ddx_ddx = self.temporal_kernel(opt_point, opt_point).evaluate()
            tmpr_k_X_ddx = torch.tile(tmpr_k_X_ddx, (1, len(self.constr_dims)))
            tmpr_k_ddx_ddx = torch.tile(tmpr_k_ddx_ddx, (len(self.constr_dims), len(self.constr_dims)))

            # k = k_s * k_t
            k_X_ddx = spatio_k_X_ddx * tmpr_k_X_ddx
            k_ddx_ddx = spatio_k_ddx_ddx * tmpr_k_ddx_ddx

            v2_tilde = torch.linalg.solve(self.L, k_X_ddx)
            A2_tilde = torch.linalg.solve(self.L.T, v2_tilde).T
            B2_tilde = k_ddx_ddx - v2_tilde.T @ v2_tilde

            # mean and variance of the unconstrained GP at the given opt_point
            mu = A2_tilde @ self.centered_mean_nobatch
            cov = B2_tilde

            time_calculated = time.time() - time_calculated
            t0 = time.time()
            # calc probability
            # catch non positive definite matrix
            eig_vals = torch.linalg.eigvals(cov)
            if torch.all(eig_vals.real > 0):
                prob = 1 - mvn.cdf(np.zeros(len(mu)) - nu, mean=mu.detach(), cov=cov.detach())
            else:
                prob = 1
            time_calculated_prob = time.time() - t0
            return prob, 0, time_calculated, time_calculated_prob

            # for next locations k > 1
        else:

            # reshape and convert
            chosen_locations = torch.reshape(torch.stack(chosen_locations),
                                             (len(chosen_locations), self.spatio_dims + 1))

            # calculate kernel matrices
            k_X_ddx = spatio_scale * self.spatio_kernel.base_kernel.construct_k_x1_ddx2(
                self.train_inputs[0],
                opt_point)
            k_X_dds = spatio_scale * self.spatio_kernel.base_kernel.construct_k_x1_ddx2(
                self.train_inputs[0],
                chosen_locations)
            k_dds_dds = spatio_scale * self.spatio_kernel.base_kernel.construct_k_ddx1_ddx2(
                chosen_locations,
                chosen_locations)
            k_ddx_ddx = spatio_scale * self.spatio_kernel.base_kernel.construct_k_ddx1_ddx2(
                opt_point,
                opt_point)
            k_ddx_dds = spatio_scale * self.spatio_kernel.base_kernel.construct_k_ddx1_ddx2(
                opt_point,
                chosen_locations)

            # temporal kernel
            tmpr_k_X_ddx = self.temporal_kernel(self.train_inputs[0], opt_point).evaluate()
            tmpr_k_X_dds = self.temporal_kernel(self.train_inputs[0], chosen_locations).evaluate()
            tmpr_k_dds_dds = self.temporal_kernel(chosen_locations, chosen_locations).evaluate()
            tmpr_k_ddx_ddx = self.temporal_kernel(opt_point, opt_point).evaluate()
            tmpr_k_ddx_dds = self.temporal_kernel(opt_point, chosen_locations).evaluate()
            tmpr_k_X_ddx = torch.tile(tmpr_k_X_ddx, (1, len(self.constr_dims)))
            tmpr_k_X_dds = torch.tile(tmpr_k_X_dds, (1, len(self.constr_dims)))
            tmpr_k_dds_dds = torch.tile(tmpr_k_dds_dds, (len(self.constr_dims), len(self.constr_dims)))
            tmpr_k_ddx_ddx = torch.tile(tmpr_k_ddx_ddx, (len(self.constr_dims), len(self.constr_dims)))
            tmpr_k_ddx_dds = torch.tile(tmpr_k_ddx_dds, (len(self.constr_dims), len(self.constr_dims)))

            # k = k_s * k_t
            k_X_ddx = k_X_ddx * tmpr_k_X_ddx
            k_X_dds = k_X_dds * tmpr_k_X_dds
            k_dds_dds = k_dds_dds * tmpr_k_dds_dds
            k_ddx_ddx = k_ddx_ddx * tmpr_k_ddx_ddx
            k_ddx_dds = k_ddx_dds * tmpr_k_ddx_dds

            # factors as in (AGRELL)
            v1 = torch.linalg.solve(self.L, k_X_dds)
            A1 = torch.linalg.solve(self.L.T, v1).T
            B1 = k_dds_dds + self.location_noise * torch.eye(k_dds_dds.shape[0]) - v1.T @ v1

            # catch non positive definite matrix
            eig_vals_B1 = torch.linalg.eigvals(B1)
            if not torch.all(eig_vals_B1.real > 0):
                prob = 1
                return prob, 0., time.time() - time_calculated, 0.

            L1 = torch.linalg.cholesky(B1)
            v2_tilde = torch.linalg.solve(self.L, k_X_ddx)

            B2_tilde = k_ddx_ddx - v2_tilde.T @ v2_tilde
            B3_tilde = k_ddx_dds - v2_tilde.T @ v1

            v3_tilde = torch.linalg.solve(L1, B3_tilde.T)
            A2_tilde = torch.linalg.solve(self.L.T, v2_tilde).T

            A_tilde = torch.linalg.solve(L1.T, v3_tilde).T
            B_tilde = A2_tilde - A_tilde @ A1

            # mu and cov of the truncated normal
            trunc_mu = A1 @ self.centered_mean_nobatch
            trunc_cov = B1

            if use_approximation:
                # only implemented for special case where lb = 0 and ub = inf
                s = np.sqrt(np.diagonal(trunc_cov.detach()))
                a_tilde = (0 - trunc_mu.detach()) / s

                factor = norm.pdf(a_tilde) / 0.5

                # set up conditional expectation (converting to numpy in the process)
                trunc_mean_approx = trunc_mu.detach() + s * factor
                trunc_cov_appprox = np.diag(s ** 2 * (1 + 0 - factor ** 2))
                mu = A_tilde.detach().numpy() @ trunc_mean_approx.detach().numpy()
                mu += B_tilde.detach().numpy() @ self.centered_mean_nobatch.detach().numpy()
                sigma_tilde = B2_tilde.detach().numpy() - v3_tilde.T.detach().numpy() @ v3_tilde.detach().numpy()
                cov = sigma_tilde + A_tilde.detach().numpy() @ trunc_cov_appprox @ A_tilde.T.detach().numpy()
                time_calculated = time.time() - time_calculated
                time_sampled = 0.
                eig_vals = torch.linalg.eigvals(torch.from_numpy(cov))
                time_calc_prob = time.time()
                # calc probability
                if torch.all(eig_vals.real > 0):
                    prob = 1 - mvn.cdf(np.zeros(len(mu)) - nu, mean=mu, cov=cov)
                else:
                    prob = 1
                time_calc_prob = time.time() - time_calc_prob
            else:
                time_calculated = time.time() - time_calculated
                time_sampled = time.time()
                C = self.sample_from_truncated_norm(trunc_mu, trunc_cov, n_samples=n_samples,
                                                    sampling_method='minimax_tilting').numpy()

                time_sampled = time.time() - time_sampled
                time_calc_prob = time.time()
                prob = 0
                for Cj in C:
                    mu = A_tilde.detach().numpy() @ (Cj - 0)
                    mu += B_tilde.detach().numpy() @ self.shifted_training_traget.detach().numpy()
                    cov = B2_tilde.detach().numpy() - v3_tilde.T.detach().numpy() @ v3_tilde.detach().numpy()

                    prob += 1 - mvn.cdf(np.zeros(len(mu)) - nu, mean=mu, cov=cov)
                prob = prob / n_samples
                time_calc_prob = time.time() - time_calc_prob

            # return weighted sum
            return prob, time_sampled, time_calculated, time_calc_prob

    def sample_from_truncated_norm(self, mu_trunc, cov_trunc, n_samples=1000, sampling_method='minimax_tilting'):
        ndim = mu_trunc.shape[0]

        # convert to numpy for sampling
        if isinstance(mu_trunc, Tensor):
            mu_trunc = mu_trunc.detach().numpy()
            cov_trunc = cov_trunc.detach().numpy()

        if sampling_method == 'minimax_tilting':
            lb = np.ones(ndim) * self.bounds[0]
            ub = np.ones(ndim) * self.bounds[1]

            t0 = time.time()
            print(f"Start minimax algorithm (Python implementation) for {n_samples} samples (D={ndim})...")
            tn = TruncatedMVN(mu_trunc, cov_trunc, lb, ub)
            samples = tn.sample(n_samples)
            print(f"Time taken for {n_samples} samples: {time.time() - t0:>4.2f}s.")
            return torch.from_numpy(samples)
        else:
            raise NotImplementedError

    @property
    def num_train(self):
        return len(self.train_inputs[0])

    def get_joint_distribution(self, X):
        # Concatenate the input to the training inputs
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [X]
        full_inputs = []
        batch_shape = train_inputs[0].shape[:-2]
        for train_input, input in zip(train_inputs, inputs):
            # Make sure the batch shapes agree for training/test data
            if batch_shape != train_input.shape[:-2]:
                batch_shape = _mul_broadcast_shape(batch_shape, train_input.shape[:-2])
                train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
            if batch_shape != input.shape[:-2]:
                batch_shape = _mul_broadcast_shape(batch_shape, input.shape[:-2])
                train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                input = input.expand(*batch_shape, *input.shape[-2:])
            full_inputs.append(torch.cat([train_input, input], dim=-2))

        # Get the joint distribution for training/test data
        full_output = super(ExactGP, self).__call__(*full_inputs)
        if settings.debug().on():
            if not isinstance(full_output, MultivariateNormal):
                raise RuntimeError("ExactGP.forward must return a MultivariateNormal!")
        return full_output.loc, full_output.lazy_covariance_matrix
