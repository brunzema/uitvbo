import time
import torch
import gpytorch
from gpytorch.utils.errors import NotPSDError
import numpy as np
from botorch import fit_gpytorch_model
from botorch.optim import optimize_acqf
from src.costum_acqusition_functions import CostumizedUpperConfidenceBound, CostumizedPosteriorMean, \
    CostumizedExpectedImprovement, CostumizedProbabilityOfImprovement, CostumizedqUpperConfidenceBound
from src.models.constrained_spatiotemporal_model import ConstrainedGPThroughTime
from utils.stats_utils import calculate_log_likelihood
from scipy.stats import binned_statistic_dd
import pickle


class TVBO_Base:
    def __init__(self,
                 objective_function_options,
                 model_options,
                 post_processing_options,
                 maximze_acq=False,
                 learn_hyper_parameters=True,
                 verbose=True,
                 approximation_type=None,
                 approximation_factor=20,
                 dtype=torch.float,
                 normalize=True,
                 withinmodelcomparison=False,
                 add_noise=True,
                 delta=1.5,
                 **kwargs):

        # torch options
        self.dtype = dtype
        self.verbose = verbose
        self.approximation_type = approximation_type
        self.withinmodelcomparison = withinmodelcomparison

        # algorithm options
        self.time_horizon = None
        self.stable_list = []
        self.maximze_acq = maximze_acq
        self.learn_hyper_parameters = learn_hyper_parameters
        self.approximation_factor = approximation_factor
        self.normalize = normalize
        self.gp_ucb_beta = 1
        self.data_mean = None
        self.data_stdv = None
        self.add_noise = add_noise
        self.delta = delta

        # objective function stuff
        self.objective_function_options = objective_function_options
        self.objective_function = objective_function_options['objective_function']
        self.spatio_dimensions = objective_function_options['spatio_dimensions']
        self.noise_lvl = objective_function_options['noise_lvl']
        self.scaling_factors = objective_function_options['scaling_factors']
        self.feasible_set = objective_function_options['feasible_set'] / self.scaling_factors
        if 'initial_feasible_set' in objective_function_options.keys():
            self.initial_feasible_set = objective_function_options['initial_feasible_set'] / self.scaling_factors
        else:
            self.initial_feasible_set = self.feasible_set

        # model options
        self.model_options = model_options
        self.nr_samples = model_options['nr_samples']
        self.xv_points_per_dim = model_options['xv_points_per_dim']
        self.constrained_dimensions = model_options['constrained_dimensions']
        self.evaluate_constrained = bool(self.constrained_dimensions)
        self.forgetting_type = model_options['forgetting_type']
        self.forgetting_factor = model_options['forgetting_factor']
        self.prior_mean = model_options['prior_mean']
        self.lengthscale_constraint = model_options['lengthscale_constraint']
        self.lengthscale_hyperprior = model_options['lengthscale_hyperprior']
        if 'outputscale_constraint_spatio' in model_options:
            self.outputscale_constraint_spatio = model_options['outputscale_constraint_spatio']
            self.outputscale_hyperprior_spatio = model_options['outputscale_hyperprior_spatio']
        else:
            self.outputscale_constraint_spatio = None
            self.outputscale_hyperprior_spatio = None
        self.truncation_bounds = model_options['truncation_bounds']

        # post processing
        self.post_processing_options = post_processing_options
        if 'plot_timestep' in self.post_processing_options:
            self.plot_timestep = self.post_processing_options['plot_timestep']
            self.plot_function = self.post_processing_options['plot_function']
            self.path_to_safe_imgs = self.post_processing_options['path_to_safe_imgs']
        else:
            self.plot_timestep = False

        if 'evaluate_likelihood' in self.post_processing_options:
            self.evaluate_likelihood = self.post_processing_options['evaluate_likelihood']
        else:
            self.evaluate_likelihood = False


class TimeVaryingBOModel(TVBO_Base):

    def run_TVBO(self, n_initial_points, time_horizon, safe_name='', trial=0):
        self.time_horizon = time_horizon
        chosen_query = []
        iter_lengthscales = []
        loglikelihoods = []
        min_trajectory = []

        print(f'Initialization')
        train_x, train_y, t_remain = self.generate_initial_data(n_initial_points, trial)
        current_time = t_remain[0] - 1
        mll, model = self.initialize_model(train_x, train_y, current_time=current_time)

        # save inital set of training data / to save, rescale to original
        start_x, start_y = train_x.clone(), train_y.clone()
        observed_fx = train_y.clone()

        if self.learn_hyper_parameters:
            fit_gpytorch_model(mll)

        iter_lengthscales.append(model.spatio_kernel.base_kernel.lengthscale.clone().detach().numpy())

        # for the start, distribute the points equidistant across the feasible set
        xv_points = self.get_virtual_obs_points([], [], current_time, initial_points=True)

        # set up options for the posterior (constrained/not constrained, etc.)
        posterior_options = {'evaluate_constrained': False,
                             'virtual_oberservation_points': xv_points,
                             'nr_samples': self.nr_samples}
        acqf = self.get_acquisition_function(model, posterior_options, [], acq='PM')
        current_optimum, _ = self.optimize_acqf_and_get_observation(acqf, self.feasible_set, current_time,
                                                                    [model, posterior_options])
        posterior_options['evaluate_constrained'] = self.evaluate_constrained

        # reset samples, as first sampling was only needed for the initialization and current optimum
        model.reset_samples()
        model.reset_factors()

        for t in t_remain:

            self.gp_ucb_beta = 2  # 0.8 * torch.log10(4 * t)  # as in Bugunovic2016/Wang2021

            if self.verbose:
                print('')
                print(f'Timestep {t:4.0f} of Trail {trial:2.0f}.')
                print(f'Parameter name: noise value = {model.likelihood.noise_covar.noise.item():0.3f}')
                print(f'Parameter name: outputscale value = {model.spatio_kernel.outputscale.item():0.3f}')
                lengthscales = model.spatio_kernel.base_kernel.lengthscale[0].clone().detach()
                for i, lengthscale in enumerate(lengthscales):
                    print(f'Parameter name: lengthscale {i} value = {lengthscale.item():0.3f}')

            t0 = time.time()
            xv_points = self.get_virtual_obs_points(model, current_optimum, t)

            # update posterior options
            posterior_options['virtual_oberservation_points'] = xv_points

            # get current aquisition bounds
            if self.evaluate_constrained:
                spatio_acq_bounds = self.get_spatio_aquisition_bounds(model, current_optimum)
            else:
                spatio_acq_bounds = self.feasible_set

            # get updated optimum
            acqf_PM = self.get_acquisition_function(model, posterior_options, [], acq='PM')
            current_optimum, _ = self.optimize_acqf_and_get_observation(acqf_PM, spatio_acq_bounds, t,
                                                                        [model, posterior_options])
            min_trajectory.append((current_optimum[0, :self.spatio_dimensions].clone()).numpy())

            # get query
            if self.approximation_type == 'reset' and t % self.approximation_factor == 0:
                new_x, new_y, _ = self.generate_initial_data(n_training_points=1, seed=t, start_time=t)
            else:
                # get query and observation from UCB/EI/PI
                acqf = self.get_acquisition_function(model, posterior_options, current_optimum)
                new_x, new_y = self.optimize_acqf_and_get_observation(acqf, spatio_acq_bounds, t,
                                                                      [model, posterior_options], get_query=True)

            # add new training points
            train_x = torch.cat((train_x, new_x), dim=0)
            train_y = torch.cat((train_y, new_y))

            # add observations
            chosen_query.append((new_x[0, :self.spatio_dimensions].clone()).numpy())
            observed_fx = torch.cat((observed_fx, new_y))

            results_objective_function_options = self.objective_function_options.copy()
            results_objective_function_options.pop('objective_function')
            results = {'trajectory': np.array(chosen_query),
                       'f_of_x': np.array(observed_fx),
                       'start_x': start_x,
                       'start_y': start_y,
                       'stable': self.stable_list,
                       'inital_data_mean_and_stdv': (self.data_mean, self.data_stdv),
                       'scaling_factors': self.scaling_factors.numpy(),
                       'min_trajectory': np.array(min_trajectory),
                       'lengthscales': np.array(iter_lengthscales),
                       'settings': {'modeloptions': self.model_options,
                                    'objectivefunctionoptions': results_objective_function_options}}

            # post processing
            if self.plot_timestep:
                name = self.path_to_safe_imgs + f'/{safe_name}_{t:03}.png'
                self.plot_function(self.objective_function, name, model, t, train_x, posterior_options,
                                   self.feasible_set, spatio_acq_bounds)

            if self.evaluate_likelihood:
                t0_loglikelihood = time.time()
                loglikelihood = self.calculate_loglikelihood(model, t, posterior_options)
                loglikelihoods.append(loglikelihood)
                results['loglikelihood'] = np.array(loglikelihoods)
                if self.verbose:
                    print(f'Time for calculating the loglikelihood: {time.time() - t0_loglikelihood:0.3f}s.')

            if self.evaluate_constrained:
                results['virtual_obs_points'] = model.optimized_VOP

            # safe results at each timestep to reduce memory usage
            forgetting_factor_name = f'{self.forgetting_factor}'.replace('.', '_')
            with open(f'results_{safe_name}_{forgetting_factor_name}_{trial}.pickle', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # prepare/select data for the next iteration
            if self.approximation_type:
                train_x, train_y = self.data_selection(train_x, train_y)

            # initialize updated model for next iteration
            mll, model = self.initialize_model(train_x, train_y, current_time=t + 1,
                                               state_dict=model.state_dict())
            if self.learn_hyper_parameters:
                fit_gpytorch_model(mll)
            iter_lengthscales.append(model.spatio_kernel.base_kernel.lengthscale.clone().detach().numpy())

            if self.verbose:
                print(f'Time taken for timestep {t:4.0f}: {time.time() - t0:0.3f}s.')

    def generate_initial_data(self, n_training_points, seed, start_time=0):
        inital_data = []
        for i in range(self.spatio_dimensions):
            x_i = torch.linspace(self.initial_feasible_set[0, i], self.initial_feasible_set[1, i],
                                 100 + n_training_points)

            # shuffling using numpy
            idx_x_i = np.arange(100 + n_training_points)
            np.random.seed(seed + i)
            np.random.shuffle(idx_x_i)

            train_x_i = x_i[idx_x_i[:n_training_points]]
            inital_data.append(train_x_i)

        # add temporal dimension and create train_x
        t = torch.arange(start_time, start_time + n_training_points)
        inital_data.append(t)
        train_x = torch.stack(inital_data, dim=1)

        # create training data
        train_y = self.objective_function(train_x[:, 0:self.spatio_dimensions] * self.scaling_factors,
                                          train_x[:, -1])
        if self.add_noise:
            train_y += torch.normal(mean=0, std=self.noise_lvl, size=train_y.size())  # noisy measurement

        # scale data to zero mean and stdv of 1
        if start_time == 0:
            if self.normalize:
                self.data_mean = torch.mean(train_y)
                self.data_stdv = torch.std(train_y)
            else:
                self.data_mean = 0
                self.data_stdv = 1

        scaled_train_y = (train_y - self.data_mean) / self.data_stdv
        self.stable_list = [True for i in range(n_training_points)]  # initalized controllers are stable

        t_remain = torch.arange(self.time_horizon)[n_training_points:]
        return train_x, scaled_train_y, t_remain

    def initialize_model(self, train_x, train_y, current_time=0, state_dict=None):
        # initialize gp model with the specifications from 'model_options'
        model = ConstrainedGPThroughTime(
            train_x,
            train_y,
            constrained_dims=self.constrained_dimensions,
            type_of_forgetting=self.forgetting_type,
            lengthscale_constraint=self.lengthscale_constraint,
            lengthscale_hyperprior=self.lengthscale_hyperprior,
            outputscale_constraint_spatio=self.outputscale_constraint_spatio,
            outputscale_hyperprior_spatio=self.outputscale_hyperprior_spatio,
            outputscale_constraint_temporal=self.outputscale_constraint_spatio,
            prior_mean=self.prior_mean,
            forgetting_factor=self.forgetting_factor, )

        # set options, freeze noise lvl
        model.likelihood.noise_covar.noise = (self.noise_lvl / self.data_stdv) ** 2
        model.likelihood.noise_covar.raw_noise.requires_grad = False
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)

        if self.forgetting_type == 'B2P_OU' and self.withinmodelcomparison:
            prior_outputscale = self.outputscale_hyperprior_spatio.mean
            model.spatio_kernel.outputscale = prior_outputscale + self.forgetting_factor * current_time

        if self.withinmodelcomparison:
            model.spatio_kernel.raw_outputscale.requires_grad = False

        # update truncation bounds
        model.bounds = self.truncation_bounds
        return mll, model

    def get_acquisition_function(self, model, posterior_options, current_optimum, acq='UCB'):
        if acq == 'UCB':
            acq_function = CostumizedUpperConfidenceBound(
                model=model,
                beta=self.gp_ucb_beta,
                posterior_options=posterior_options,
                maximize=self.maximze_acq,
            )
        elif acq == 'PM':
            acq_function = CostumizedPosteriorMean(
                model=model,
                posterior_options=posterior_options,
                maximize=self.maximze_acq,
            )
        elif acq == "EI":  # as defined in Renganathan2020
            acq_function = CostumizedExpectedImprovement(
                model=model,
                best_f=current_optimum[0, 0],
                posterior_options=posterior_options,
                maximize=self.maximze_acq,
            )
        elif acq == "PI":  # as defined in Renganathan2020
            acq_function = CostumizedProbabilityOfImprovement(
                model=model,
                best_f=current_optimum[0, 0],
                posterior_options=posterior_options,
                maximize=self.maximze_acq,
            )
        elif acq == 'qUCB':
            acq_function = CostumizedqUpperConfidenceBound(
                model=model,
                beta=self.gp_ucb_beta,
                posterior_options=posterior_options,
                maximize=self.maximze_acq,
            )
        else:
            raise NotImplementedError
        return acq_function

    def optimize_acqf_and_get_observation(self, acqf, spatio_bounds, t, model_posterior, get_query=False):
        bounds = torch.cat((spatio_bounds, torch.ones(2, 1) * t), dim=1)

        t0_acqf = time.time()
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=100,  # used for intialization heuristic
            options={}, )

        if self.verbose:
            print(f'Time for optimizing aquisition functions: {time.time() - t0_acqf:0.3f}s.')

        # observe new values
        new_x = candidates.detach()
        new_y = self.objective_function(new_x[:, 0:self.spatio_dimensions] * self.scaling_factors,
                                        new_x[0, -1].unsqueeze(0))

        # TODO: Adapt to general case (upper bound on f(t)).
        if new_y.item() > 100.:  # if controller is unstable, take f = \mu + 3*\sigma
            model = model_posterior[0]
            posterior_options = model_posterior[1]
            observed_pred = model.posterior(
                new_x,
                evaluate_constrained=posterior_options['evaluate_constrained'],
                virtual_oberservation_points=posterior_options['virtual_oberservation_points'],
                nr_samples=posterior_options['nr_samples'])
            mean = observed_pred.mean.clone().detach().reshape(-1)
            std = torch.sqrt(observed_pred.variance.clone().detach()).reshape(-1)
            new_y = mean + 3 * std
            if get_query:
                self.stable_list.append(False)
        else:
            if self.add_noise:
                new_y += torch.normal(mean=0., std=self.noise_lvl, size=new_y.size())
            new_y = (new_y - self.data_mean) / self.data_stdv
            if get_query:
                self.stable_list.append(True)
        return new_x, new_y

    def get_virtual_obs_points(self, model, current_optimum, t, initial_points=False):
        x_i_vectors = []
        if initial_points:  # we don't know the current optimum
            for x_i in range(self.spatio_dimensions):
                lb_i, ub_i = self.feasible_set[0, x_i], self.feasible_set[1, x_i]
                support_vec = torch.linspace(lb_i, ub_i, self.xv_points_per_dim)
                x_i_vectors.append(support_vec)
        else:  # we know the current optimum
            model_lengthscales = model.spatio_kernel.base_kernel.lengthscale[0].clone().detach()
            for x_i in range(self.spatio_dimensions):
                # bounds as 2 lengthscales "left and right" in each dimension with the corresonding lengthscale
                lb_i = current_optimum[0, x_i] - self.delta * model_lengthscales[x_i]
                ub_i = current_optimum[0, x_i] + self.delta * model_lengthscales[x_i]
                support_vec = torch.linspace(lb_i.item(), ub_i.item(), self.xv_points_per_dim)
                x_i_vectors.append(support_vec)

        # combine dimensions and add time
        if self.spatio_dimensions > 1:
            xv_x = torch.stack(torch.meshgrid(x_i_vectors), dim=self.spatio_dimensions).reshape(-1,
                                                                                                self.spatio_dimensions)
        else:
            xv_x = x_i_vectors[0].reshape(-1, self.spatio_dimensions)

        xv_t = torch.ones_like(xv_x)[:, 0].reshape(-1, 1) * t
        xv = torch.cat((xv_x, xv_t), dim=1)
        return xv

    def get_spatio_aquisition_bounds(self, model, current_optimum):
        acq_bounds = torch.empty(2, 0)
        model_lengthscales = model.spatio_kernel.base_kernel.lengthscale[0].clone().detach()
        for x_i in range(self.spatio_dimensions):
            # check lower bound
            proposed_lb = current_optimum[0, x_i] - model_lengthscales[x_i]
            feasible_lb = self.feasible_set[0, x_i]
            lb = proposed_lb if proposed_lb > feasible_lb else feasible_lb

            proposed_ub = current_optimum[0, x_i] + model_lengthscales[x_i]
            feasible_ub = self.feasible_set[1, x_i]
            ub = proposed_ub if proposed_ub < feasible_ub else feasible_ub
            acq_bounds = torch.cat((acq_bounds, torch.tensor([[lb], [ub]])), dim=1)
        return acq_bounds

    def calculate_loglikelihood(self, model, t, posterior_options, resolution=50):
        try:
            x_test = []
            for x_i in range(self.spatio_dimensions):
                xi_test = torch.linspace(self.feasible_set[0, x_i], self.feasible_set[1, x_i], resolution)
                x_test.append(xi_test)

            if self.spatio_dimensions > 1:
                x_test = torch.stack(torch.meshgrid(x_test), dim=self.spatio_dimensions).reshape(-1,
                                                                                                 self.spatio_dimensions)
                t_test = torch.ones_like(x_test)[:, 0].reshape(-1, 1) * t
            else:
                x_test = x_test[0].reshape(-1, self.spatio_dimensions)
                t_test = torch.ones_like(x_test) * t

            x_test = torch.cat((x_test, t_test), dim=1)

            y_test = self.objective_function(x_test[:, 0:self.spatio_dimensions], x_test[:, -1]).reshape(-1, 1)
            observed_pred = model.posterior(
                x_test,
                evaluate_constrained=posterior_options['evaluate_constrained'],
                virtual_oberservation_points=posterior_options['virtual_oberservation_points'],
                nr_samples=posterior_options['nr_samples'])

            # rescale based on the mean and variance which was accumulated
            mean = (observed_pred.mean.detach() * self.data_stdv + self.data_mean).numpy()
            var = (observed_pred.variance.detach() * self.data_stdv + self.data_mean).numpy()
            var = np.where(var <= 0., 0.000001, var)
            loglikelihood = calculate_log_likelihood(mean.squeeze(-1), var.squeeze(-1), y_test.squeeze(-1))
        except NotPSDError:
            print('Warning: There has been a "NotPSDError" while calculating the loglikelihood.')
            loglikelihood = None

        return loglikelihood

    def data_selection(self, x_train, y_train):
        # sort data (decreasing time)
        sort_index = torch.argsort(x_train[:, -1], descending=True)
        sorted_x_train = x_train[sort_index, :]
        sorted_y_train = y_train[sort_index]

        # binning
        if self.approximation_type == 'binning':
            n_bin_per_dim = self.approximation_factor

            # for binning keep last 10 samples, regardless of the bin!
            x_train_new, remaining_x_train = sorted_x_train[:10, :], sorted_x_train[10:, :]
            y_train_new, remaining_y_train = sorted_y_train[:10], sorted_y_train[10:]

            bin_edges = []
            for x_i in range(self.spatio_dimensions):
                xi_bins = np.linspace(self.feasible_set[0, x_i], self.feasible_set[1, x_i], n_bin_per_dim + 1)
                bin_edges.append(xi_bins)

            # CARE: number of bins scales exponentially,
            # +2 since index on the borders are interpreted as outliers!
            bins = np.arange((n_bin_per_dim + 2) ** self.spatio_dimensions)

            # remove edges, TODO: SUPER UGLY
            if self.spatio_dimensions == 1:
                bins = bins[1:-1]
            elif self.spatio_dimensions == 2:
                bins = bins.reshape(n_bin_per_dim + 2, n_bin_per_dim + 2)
                bins = bins[1:-1, 1:-1].reshape(-1)  # remove edges
            elif self.spatio_dimensions == 3:
                bins = bins.reshape(n_bin_per_dim + 2, n_bin_per_dim + 2, n_bin_per_dim + 2)
                bins = bins[1:-1, 1:-1, 1:-1].reshape(-1)
            elif self.spatio_dimensions == 4:
                bins = bins.reshape(n_bin_per_dim + 2, n_bin_per_dim + 2, n_bin_per_dim + 2, n_bin_per_dim + 2)
                bins = bins[1:-1, 1:-1, 1:-1, 1:-1].reshape(-1)

            bin_res = binned_statistic_dd(sorted_x_train[:, 0:self.spatio_dimensions].numpy(),
                                          None,
                                          bins=bin_edges,
                                          statistic='count')
            bin_idx = bin_res.binnumber
            available_bins = set(np.setxor1d(bins, bin_idx[:10]))
            for i, (data_x, data_y) in enumerate(zip(remaining_x_train, remaining_y_train), start=10):
                if not available_bins:  # if all bins have been filled
                    break

                if bin_idx[i] in available_bins:  # if bin has not been filled, fill bin with current data point
                    x_train_new = torch.cat((x_train_new, data_x.unsqueeze(0)), dim=0)
                    y_train_new = torch.cat((y_train_new, data_y.unsqueeze(0)), dim=0)
                    available_bins.remove(bin_idx[i])

            print(f'Still {len(available_bins)} bins not filled. Currently {x_train_new.shape[0]} data points.')

        # apply sliding window for B2P forgetting
        elif self.approximation_type == 'sliding_window':
            sw_number = self.approximation_factor
            if len(sorted_x_train) > sw_number:
                x_train_new = sorted_x_train[:sw_number, :]
                y_train_new = sorted_y_train[:sw_number]
                print(f'Data section using sliding window of {sw_number}.')
            else:
                x_train_new = sorted_x_train
                y_train_new = sorted_y_train

        elif self.approximation_type == 'reset':
            reset_nr = self.approximation_factor
            t = sorted_x_train[0, -1].item()
            if t % reset_nr == 0:
                x_train_new = sorted_x_train[:1, :]
                y_train_new = sorted_y_train[:1]
                print(f'Data section using reset of {reset_nr}.')
            else:
                x_train_new = sorted_x_train
                y_train_new = sorted_y_train
        else:
            raise RuntimeError

        return x_train_new, y_train_new
