#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Analytic Acquisition Functions that evaluate the posterior without performing
Monte-Carlo sampling.
"""

from __future__ import annotations

from abc import ABC
from typing import Dict, Optional, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective
from botorch.exceptions import UnsupportedError
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal


class CostumizedAnalyticAcquisitionFunction(AcquisitionFunction, ABC):
    r"""Base class for analytic acquisition functions.

    Copy paste from the documentation but changed the _getposterior function
    to pass options to the posterior call of the model"""

    def __init__(
            self, model: Model,
            objective: Optional[ScalarizedObjective] = None,
            posterior_options: Dict = None,
    ) -> None:
        r"""Base constructor for analytic acquisition functions.

        Args:
            model: A fitted single-outcome model.
            objective: A ScalarizedObjective (optional).
        """
        super().__init__(model=model)
        if objective is None:
            if model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify an objective when using a multi-output model."
                )
        elif not isinstance(objective, ScalarizedObjective):
            raise UnsupportedError(
                "Only objectives of type ScalarizedObjective are supported for "
                "analytic acquisition functions."
            )
        self.objective = objective
        self.posterior_options = posterior_options

    def _get_posterior(self, X: Tensor) -> Posterior:
        r"""Compute the posterior at the input candidate set X.

        Applies the objective if provided.

        Args:
            X: The input candidate set.

        Returns:
            The posterior at X. If a ScalarizedObjective is defined, this
            posterior can be single-output even if the underlying model is a
            multi-output model.
        """
        posterior = self.model.posterior(X=X, observation_noise=None, **self.posterior_options)
        if self.objective is not None:
            # Unlike MCAcquisitionObjective (which transform samples), this
            # transforms the posterior
            posterior = self.objective(posterior)
        return posterior

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        raise UnsupportedError(
            "Analytic acquisition functions do not account for X_pending yet."
        )


class CostumizedExpectedImprovement(CostumizedAnalyticAcquisitionFunction):
    r"""Single-outcome Expected Improvement (analytic).

    Computes classic Expected Improvement over the current best observed value,
    using the analytic formula for a Normal posterior distribution. Unlike the
    MC-based acquisition functions, this relies on the posterior at single test
    point being Gaussian (and require the posterior to implement `mean` and
    `variance` properties). Only supports the case of `q=1`. The model must be
    single-outcome.

    `EI(x) = E(max(y - best_f, 0)), y ~ f(x)`
    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            posterior_options: Dict = None,
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective, posterior_options=posterior_options)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `b1 x ... bk`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self._get_posterior(X)
        mean = posterior.mean
        # deal with batch evaluation and broadcasting
        view_shape = mean.shape[:-2] if mean.dim() >= X.dim() else X.shape[:-2]
        mean = mean.view(view_shape)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei


class CostumizedPosteriorMean(CostumizedAnalyticAcquisitionFunction):
    r"""Single-outcome Posterior Mean.

    Only supports the case of q=1. Requires the model's posterior to have a
    `mean` property. The model must be single-outcome.
    """

    def __init__(
            self,
            model: Model,
            objective: Optional[ScalarizedObjective] = None,
            posterior_options: Dict = None,
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective, posterior_options=posterior_options)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior mean on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Posterior Mean values at the given design
            points `X`.
        """
        posterior = self._get_posterior(X=X)
        mean = posterior.mean.view(X.shape[:-2])
        if self.maximize:
            return mean
        else:
            return -mean


class CostumizedProbabilityOfImprovement(CostumizedAnalyticAcquisitionFunction):
    r"""Single-outcome Probability of Improvement.

    Probability of improvment over the current best observed value, computed
    using the analytic formula under a Normal posterior distribution. Only
    supports the case of q=1. Requires the posterior to be Gaussian. The model
    must be single-outcome.

    `PI(x) = P(y >= best_f), y ~ f(x)`
    """

    def __init__(
            self,
            model: Model,
            best_f: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            posterior_options: Dict = None,
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome analytic Probability of Improvement.

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective, posterior_options=posterior_options)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Probability of Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim tensor of Probability of Improvement values at the given
            design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self._get_posterior(X=X)
        mean, sigma = posterior.mean, posterior.variance.sqrt()
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        sigma = posterior.variance.sqrt().clamp_min(1e-9).view(batch_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        return normal.cdf(u)


class CostumizedUpperConfidenceBound(CostumizedAnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.
    """

    def __init__(
            self,
            model: Model,
            beta: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            posterior_options: Dict = None,
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective, posterior_options=posterior_options)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta = self.beta.to(X)
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        variance = posterior.variance.view(batch_shape)
        delta = (self.beta.expand_as(mean) * variance).sqrt()
        if self.maximize:
            return mean + delta
        else:
            return -mean + delta


# TODO:hier weiter
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.acquisition.objective import IdentityMCObjective, MCAcquisitionObjective
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)


class CostumMCAcquisitionFunction(AcquisitionFunction, ABC):
    r"""Abstract base class for Monte-Carlo based batch acquisition functions."""

    def __init__(
            self,
            model: Model,
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            X_pending: Optional[Tensor] = None,
            posterior_options: Dict = None
    ) -> None:
        r"""Constructor for the MCAcquisitionFunction base class.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)`.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
        """
        super().__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)
        self.add_module("sampler", sampler)
        if objective is None:
            if model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify an objective when using a multi-output model."
                )
            objective = IdentityMCObjective()
        elif not isinstance(objective, MCAcquisitionObjective):
            raise UnsupportedError(
                "Only objectives of type MCAcquisitionObjective are supported for "
                "MC acquisition functions."
            )
        self.add_module("objective", objective)
        self.set_X_pending(X_pending)
        self.objective = objective
        self.posterior_options = posterior_options
        self.posterior_options['return_samples'] = True

    def forward(self, X: Tensor) -> Tensor:
        r"""Takes in a `batch_shape x q x d` X Tensor of t-batches with `q` `d`-dim
        design points each, and returns a Tensor with shape `batch_shape'`, where
        `batch_shape'` is the broadcasted batch shape of model and input `X`. Should
        utilize the result of `set_X_pending` as needed to account for pending function
        evaluations.
        """
        pass  # pragma: no cover


class CostumizedqUpperConfidenceBound(CostumMCAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.
    """

    def __init__(
            self,
            model: Model,
            beta: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            posterior_options: Dict = None,
            maximize: bool = True,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, objective=objective, posterior_options=posterior_options)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound on the candidate set `X`.

        Args:
            X: A `batch_sahpe x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        samples = self.model.posterior(X=X, observation_noise=None, **self.posterior_options)
        # samples = self.sampler(posterior)
        # obj = self.objective(samples, X=X)
        obj = torch.permute(samples, (2, 0, 1))
        # obj = self.objective(samples, X=X)
        mean = obj.mean(dim=0)
        if self.maximize:
            ucb_samples = mean + self.beta * (obj - mean).abs()
            return ucb_samples.max(dim=-1)[0].mean(dim=0)
        else:
            ucb_samples = mean - self.beta * (obj - mean).abs()
            return ucb_samples.min(dim=-1)[0].mean(dim=0)
