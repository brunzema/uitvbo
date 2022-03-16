from src.TVBO import TimeVaryingBOModel
import gpytorch
import torch
from src.objective_functions_LQR import lqr_objective_function_2D

objective_function_options = {'objective_function': lqr_objective_function_2D,
                              'spatio_dimensions': 2,
                              'noise_lvl': 0.005,
                              'feasible_set': torch.tensor([[-62.5, -5], [-12.5, -1]],
                                                           dtype=torch.float),
                              'initial_feasible_set': torch.tensor([[-50, -4], [-25, -2]],
                                                                   dtype=torch.float),
                              'scaling_factors': torch.tensor([3, 1 / 4])}

# UI -> UI-TVBO, B2P_OU -> TV-GP-UCB
variations = [{'method': 'UI', 'forgetting_factor': 0.03, 'constrained_dims': [0, 1]},
              {'method': 'UI', 'forgetting_factor': 0.03, 'constrained_dims': []},
              {'method': 'B2P_OU', 'forgetting_factor': 0.03, 'constrained_dims': [0, 1]},
              {'method': 'B2P_OU', 'forgetting_factor': 0.03, 'constrained_dims': []}, ]

for variation in variations:
    method = variation['method']
    factor = variation['forgetting_factor']
    constrained_dims = variation['constrained_dims']

    model_options = {'nr_samples': 10000,
                     'constrained_dimensions': constrained_dims,
                     'xv_points_per_dim': 4,
                     'forgetting_type': method,
                     'forgetting_factor': factor,
                     'prior_mean': 0.,
                     'lengthscale_constraint': gpytorch.constraints.Interval(0.5, 6),
                     'lengthscale_hyperprior': gpytorch.priors.GammaPrior(6, 1 / 0.3),
                     'outputscale_constraint_spatio': gpytorch.constraints.Interval(0, 20),
                     'outputscale_hyperprior_spatio': None,
                     'truncation_bounds': [0, 2]}

    tvbo_model = TimeVaryingBOModel(objective_function_options=objective_function_options,
                                    model_options=model_options,
                                    post_processing_options={},
                                    approximation_type=None,  # 'binning'. 'sliding_window', 'reset'
                                    approximation_factor=None,
                                    add_noise=False,  # noise is added during the simulation of the pendulum
                                    delta=1.2)

    if constrained_dims:
        string = 'constrained'
    else:
        string = 'unconstrained'

    NAME = f'{method}_2DLQR_{string}_forgetting_factor_{factor}'.replace('.', '_')

    trials = 25
    for trial in range(1, trials + 1):
        tvbo_model.run_TVBO(n_initial_points=30,
                            time_horizon=300,
                            safe_name=NAME,
                            trial=trial, )
