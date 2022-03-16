import time

import numpy as np
import math
from scipy import special
from scipy import optimize

EPS = 10e-15


class TruncatedMVN:
    """
    Create a normal distribution :math:`X  \sim N ({\mu}, {\Sigma})` subject to linear inequality constraints
    :math:`lb < X < ub` and sample from it using minimax tilting. Based on the MATLAB implemention by the authors
    (reference below).

    :param np.ndarray mu: (size D) mean of the normal distribution :math:`\mathbf {\mu}`.
    :param np.ndarray cov: (size D x D) covariance of the normal distribution :math:`\mathbf {\Sigma}`.
    :param np.ndarray lb: (size D) lower bound constrain of the multivariate normal distribution :math:`\mathbf lb`.
    :param np.ndarray ub: (size D) upper bound constrain of the multivariate normal distribution :math:`\mathbf ub`.

    Note that the algorithm may not work if 'cov' is close to being rank deficient.

    Reference:
    Botev, Z. I., (2016), The normal law under linear restrictions: simulation and estimation via minimax tilting,
    Journal of the Royal Statistical Society Series B, 79, issue 1, p. 125-148,

    Example:
        >>> d = 10  # dimensions
        >>>
        >>> # random mu and cov
        >>> mu = np.random.rand(d)
        >>> cov = 0.5 - np.random.rand(d ** 2).reshape((d, d))
        >>> cov = np.triu(cov)
        >>> cov += cov.T - np.diag(cov.diagonal())
        >>> cov = np.dot(cov, cov)
        >>>
        >>> # constraints
        >>> lb = np.zeros_like(mu) - 2
        >>> ub = np.ones_like(mu) * np.inf
        >>>
        >>> # create truncated normal and sample from it
        >>> n_samples = 100000
        >>> samples = TruncatedMVN(mu, cov, lb, ub).sample(n_samples)

    Reimplementation by Paul Brunzema --> https://github.com/brunzema/truncated-mvn-sampler
    """

    def __init__(self, mu, cov, lb, ub):
        self.dim = len(mu)
        if not cov.shape[0] == cov.shape[1]:
            raise RuntimeError("Covariance matrix must be of shape DxD!")
        if not (self.dim == cov.shape[0] and self.dim == len(lb) and self.dim == len(ub)):
            raise RuntimeError("Dimensions D of mean (mu), covariance matric (cov), lower bound (lb) "
                               "and upper bound (ub) must be the same!")

        self.cov = cov
        self.orig_mu = mu
        self.orig_lb = lb
        self.orig_ub = ub

        self.moved_lb = lb - mu
        self.moved_ub = ub - mu

        # permutated
        self.lb = lb - mu  # move distr./bounds to have zero mean
        self.ub = ub - mu  # move distr./bounds to have zero mean
        if np.any(self.ub <= self.lb):
            raise RuntimeError("Upper bound (ub) must be strictly greater than lower bound (lb) for all D dimensions!")

        # scaled Cholesky with zero diagonal, permutated
        self.L = np.empty_like(cov)
        self.unscaled_L = np.empty_like(cov)
        self.scaled_L = np.empty_like(cov)

        # placeholder for optimization
        self.perm = None
        self.x = None
        self.mu = None
        self.psistar = None

        # options for Gibbs sampling
        self.switch_to_gibbs_sampling = None
        self.start_gibbs_at_iteration = None

        # for numerics
        self.eps = EPS

    def sample(self, n, max_iterations=10 ** 5, switch_to_gibbs_sampling=True, iteration_gibbs_sampling=1000):
        """
        Create n samples from the truncated normal distribution.

        :param int n: Number of samples to create.
        :param int max_iterations: max number of simulation iterations.
        :param boolean switch_to_gibbs_sampling: switch to Gibbs sampling, if iteration > iteration_gibbs_sampling.
        :param int iteration_gibbs_sampling: iteration at which Gibbs sampling should start.
        :return: D x n array with the samples.
        :rtype: np.ndarray
        """
        if not isinstance(n, int):
            raise RuntimeError("Number of samples must be an integer!")

        # factors (Cholesky, etc.) only need to be computed once!
        if self.psistar is None:
            self.compute_factors()

        # start acceptance rejection sampling
        rv = np.array([], dtype=np.float32).reshape(self.dim, 0)
        accept, iteration = 0, 0
        t0 = time.time()
        while accept < n:
            logpr, Z = self.mvnrnd(n, self.mu)  # simulate n proposals
            idx = -np.log(np.random.rand(n)) > (self.psistar - logpr)  # acceptance tests
            rv = np.concatenate((rv, Z[:, idx]), axis=1)  # accumulate accepted
            accept = rv.shape[1]  # keep track of # of accepted
            iteration += 1
            if iteration % 200 == 0:
                print(f'Iteration: {iteration}, Accepted samples: {accept}. Time taken: {time.time() - t0:10.3f}s.')
                t0 = time.time()
            if iteration == 10 ** 3:
                print('Warning: Acceptance prob. smaller than 0.001.')
            if switch_to_gibbs_sampling and iteration > iteration_gibbs_sampling:

                if accept < 2:  # create initialization, if no samples where accepted so far
                    accept = 0
                    rv = np.random.multivariate_normal(np.zeros_like(self.lb), self.cov).reshape(-1, 1)
                    rv = np.where(rv < self.moved_lb, self.moved_lb + EPS * np.random.rand(), rv)
                    rv = np.where(rv > self.moved_ub, self.moved_ub - EPS * np.random.rand(), rv)
                else:
                    # rescale minimax samples - Gibbs sampling works on the unscaled covariance matrix
                    rv = rv[:, :-1]
                    rv = self.unscaled_L @ rv

                n_remaining = n - accept
                print(f'Switching to Gibbs sampling. Remaining: {n_remaining}. SAMPLES ARE NO LONGER IID!')
                t0 = time.time()
                # print('Compile Gibbs sampler...')
                # test_L, test_rv = np.array([[1, 0], [0.2, 1]]), np.array([[0.1], [0.1]])
                # _ = gibbs_sampling(test_L.astype('float32'), test_rv.astype('float32'), 2)
                # print('Finished.')
                # Z = gibbs_sampling(self.scaled_L.astype('float32'), rv.astype('float32'), n_remaining)
                Z_gibbs, overall_time, calc_time, sample_time = self.gibbs_sampling(rv.astype('float32'), n_remaining)
                print(f'Time taken for Gibbs sampling: {time.time() - t0:10.3f}s.'
                      f'Overall time: {overall_time:10.3f}s. Calculation time: {calc_time:10.3f}s.'
                      f'Sampling time: {sample_time:10.3f}s.')

                # combine, reorder, and move to original mean
                rv = np.concatenate((rv, Z_gibbs), axis=1)
                rv = rv[:, :n]
                order = self.perm.argsort(axis=0)
                rv = rv[order, :]
                rv += np.tile(self.orig_mu.reshape(self.dim, 1), (1, rv.shape[-1]))
                return rv

            elif iteration > max_iterations:
                accept = n
                rv = np.concatenate((rv, Z), axis=1)
                print('Warning: Sample is only approximately distributed.')

        # finish sampling and postprocess the samples!
        order = self.perm.argsort(axis=0)
        rv = rv[:, :n]
        rv = self.unscaled_L @ rv
        rv = rv[order, :]

        # retransfer to original mean
        rv += np.tile(self.orig_mu.reshape(self.dim, 1), (1, rv.shape[-1]))  # Z = X + mu
        return rv

    def compute_factors(self):
        # compute permutated Cholesky factor and solve optimization

        # Cholesky decomposition of matrix with permuation
        self.unscaled_L, self.perm = self.colperm()
        D = np.diag(self.unscaled_L)
        if np.any(D < self.eps):
            print('Warning: Method might fail as covariance matrix is singular!')

        # rescale
        self.scaled_L = self.unscaled_L / np.tile(D.reshape(self.dim, 1), (1, self.dim))
        self.lb = self.lb / D
        self.ub = self.ub / D

        # remove diagonal
        self.L = self.scaled_L - np.eye(self.dim)

        # get gradient/Jacobian function
        gradpsi = self.get_gradient_function()
        x0 = np.zeros(2 * (self.dim - 1))

        # find optimal tilting parameter non-linear equation solver
        sol = optimize.root(gradpsi, x0, args=(self.L, self.lb, self.ub), method='hybr', jac=True)
        if not sol.success:
            print('Warning: Method may fail as covariance matrix is close to singular!')
        self.x = sol.x[:self.dim - 1]
        self.mu = sol.x[self.dim - 1:]

        # compute psi star
        self.psistar = self.psy(self.x, self.mu)

    def reset(self):
        # reset factors -> when sampling, optimization for optimal tilting parameters is performed again

        # permutated
        self.lb = self.orig_lb - self.orig_mu  # move distr./bounds to have zero mean
        self.ub = self.orig_ub - self.orig_mu

        # scaled Cholesky with zero diagonal, permutated
        self.L = np.empty_like(self.cov)
        self.unscaled_L = np.empty_like(self.cov)

        # placeholder for optimization
        self.perm = None
        self.x = None
        self.mu = None
        self.psistar = None

    def gibbs_sampling(self, samples, n_remaining):

        cov_s = self.cov
        t0_overall = time.time()
        Z = samples

        # variance for ith element
        t0 = time.time()
        var = np.zeros_like(self.lb)
        for i in range(self.dim):
            mask = np.ones(self.dim, dtype=bool)
            mask[i] = 0
            var[i] = cov_s[i, i] - cov_s[mask, i].T @ np.linalg.solve(cov_s[mask][:, mask], cov_s[mask, i])

        calc_time = time.time() - t0
        sample_time = 0
        t0_interval = time.time()
        for j in range(n_remaining):
            last_Z = Z[:, -1].copy()
            new_Z = np.empty_like(last_Z)
            for i in range(self.dim):
                # create mask
                mask = np.ones(self.dim, dtype=bool)
                mask[i] = 0

                # mean for ith element
                t0 = time.time()
                mu_i = cov_s[mask, i] @ np.linalg.solve(cov_s[mask][:, mask], last_Z[mask])
                calc_time += time.time() - t0

                # stdv
                s = np.sqrt(var[i])

                # scaled bounds
                lb = np.array([(self.moved_lb[i] - mu_i) / s])
                ub = np.array([(self.moved_ub[i] - mu_i) / s])

                t0 = time.time()
                X = TruncatedMVN.trandn(lb, ub)
                new_Z[i] = mu_i + X * s
                # print(new_Z[i] > -1)
                # update dimension of last Z^j = Z^{j+1}_1 ... Z^{j+1}_{i-1}, Z^{j}_{i+1}, Z^{j}_{m}
                last_Z[i] = new_Z[i]
                sample_time += time.time() - t0
                # print('Package:', time.time() - t0)
            Z = np.concatenate((Z, new_Z.reshape(-1, 1)), axis=1)
            if j % 300 == 0 and j > 0:
                print(f'Created {j} samples. Time for Gibbs sampling so far: {time.time() - t0_interval:10.3f}s.'
                      f' Continue...')
                t0_interval = time.time()
        overall_time = time.time() - t0_overall
        return Z, overall_time, calc_time, sample_time

    def mvnrnd(self, n, mu):
        # generates the proposals from the exponentially tilted sequential importance sampling pdf
        # output:     logpr, log-likelihood of sample
        #             Z, random sample
        mu = np.append(mu, [0.])
        Z = np.zeros((self.dim, n))
        logpr = 0
        for k in range(self.dim):
            # compute matrix multiplication L @ Z
            col = self.L[k, :k] @ Z[:k, :]
            # compute limits of truncation
            tl = self.lb[k] - mu[k] - col
            tu = self.ub[k] - mu[k] - col
            # simulate N(mu,1) conditional on [tl,tu]
            Z[k, :] = mu[k] + TruncatedMVN.trandn(tl, tu)
            # update likelihood ratio
            logpr += lnNormalProb(tl, tu) + .5 * mu[k] ** 2 - mu[k] * Z[k, :]
        return logpr, Z

    @staticmethod
    def trandn(lb, ub):
        """
        Sample generator for the truncated standard multivariate normal distribution :math:`X \sim N(0,I)` s.t.
        :math:`lb<X<ub`.

        If you wish to simulate a random variable 'Z' from the non-standard Gaussian :math:`N(m,s^2)`
        conditional on :math:`lb<Z<ub`, then first simulate x=TruncatedMVN.trandn((l-m)/s,(u-m)/s) and set
        Z=m+s*x.
        Infinite values for 'ub' and 'lb' are accepted.

        :param np.ndarray lb: (size D) lower bound constrain of the normal distribution :math:`\mathbf lb`.
        :param np.ndarray ub: (size D) upper bound constrain of the normal distribution :math:`\mathbf lb`.

        :return: D samples if the truncated normal distribition x ~ N(0, I) subject to lb < x < ub.
        :rtype: np.ndarray
        """
        if not len(lb) == len(ub):
            raise RuntimeError("Lower bound (lb) and upper bound (ub) must be of the same length!")

        x = np.empty_like(lb)
        a = 0.66  # threshold used in MATLAB implementation, other threshold might speed up python3 implementation
        # three cases to consider
        # case 1: a<lb<ub
        I = lb > a
        if np.any(I):
            tl = lb[I]
            tu = ub[I]
            x[I] = TruncatedMVN.ntail(tl, tu)
        # case 2: lb<ub<-a
        J = ub < -a
        if np.any(J):
            tl = -ub[J]
            tu = -lb[J]
            x[J] = - TruncatedMVN.ntail(tl, tu)
        # case 3: otherwise use inverse transform or accept-reject
        I = ~(I | J)
        if np.any(I):
            tl = lb[I]
            tu = ub[I]
            x[I] = TruncatedMVN.tn(tl, tu)
        return x

    @staticmethod
    def tn(lb, ub, tol=2):
        # samples a column vector of length=len(lb)=len(ub) from the standard multivariate normal distribution
        # truncated over the region [lb,ub], where -a<lb<ub<a for some 'a' and lb and ub are column vectors
        # uses acceptance rejection and inverse-transform method

        sw = tol  # controls switch between methods, threshold can be tuned for maximum speed for each platform
        x = np.empty_like(lb)
        # case 1: abs(ub-lb)>tol, uses accept-reject from randn
        I = abs(ub - lb) > sw
        if np.any(I):
            tl = lb[I]
            tu = ub[I]
            x[I] = TruncatedMVN.trnd(tl, tu)

        # case 2: abs(u-l)<tol, uses inverse-transform
        I = ~I
        if np.any(I):
            tl = lb[I]
            tu = ub[I]
            pl = special.erfc(tl / np.sqrt(2)) / 2
            pu = special.erfc(tu / np.sqrt(2)) / 2
            x[I] = np.sqrt(2) * special.erfcinv(2 * (pl - (pl - pu) * np.random.rand(len(tl))))
        return x

    @staticmethod
    def trnd(lb, ub):
        # uses acceptance rejection to simulate from truncated normal
        x = np.random.randn(len(lb))  # sample normal
        test = (x < lb) | (x > ub)
        I = np.where(test)[0]
        d = len(I)
        while d > 0:  # while there are rejections
            ly = lb[I]
            uy = ub[I]
            y = np.random.randn(len(uy))  # resample
            idx = (y > ly) & (y < uy)  # accepted
            x[I[idx]] = y[idx]
            I = I[~idx]
            d = len(I)
        return x

    @staticmethod
    def ntail(lb, ub):
        # samples a column vector of length=len(lb)=len(ub) from the standard multivariate normal distribution
        # truncated over the region [lb,ub], where lb>0 and lb and ub are column vectors
        # uses acceptance-rejection from Rayleigh distr. similar to Marsaglia (1964)
        if not len(lb) == len(ub):
            raise RuntimeError("Lower bound (lb) and upper bound (ub) must be of the same length!")
        c = (lb ** 2) / 2
        n = len(lb)
        f = np.expm1(c - ub ** 2 / 2)
        x = c - np.log(1 + np.random.rand(n) * f)  # sample using Rayleigh
        # keep list of rejected
        I = np.where(np.random.rand(n) ** 2 * x > c)[0]
        d = len(I)
        while d > 0:  # while there are rejections
            cy = c[I]
            y = cy - np.log(1 + np.random.rand(d) * f[I])
            idx = (np.random.rand(d) ** 2 * y) < cy  # accepted
            x[I[idx]] = y[idx]  # store the accepted
            I = I[~idx]  # remove accepted from the list
            d = len(I)
        return np.sqrt(2 * x)  # this Rayleigh transform can be delayed till the end

    def psy(self, x, mu):
        # implements psi(x,mu); assumes scaled 'L' without diagonal
        x = np.append(x, [0.])
        mu = np.append(mu, [0.])
        c = self.L @ x
        lt = self.lb - mu - c
        ut = self.ub - mu - c
        p = np.sum(lnNormalProb(lt, ut) + 0.5 * mu ** 2 - x * mu)
        return p

    def get_gradient_function(self):
        # wrapper to avoid dependancy on 'self'

        def gradpsi(y, L, l, u):
            # implements gradient of psi(x) to find optimal exponential twisting, returns also the Jacobian
            # NOTE: assumes scaled 'L' with zero diagonal
            d = len(u)
            c = np.zeros(d)
            mu, x = c.copy(), c.copy()
            x[0:d - 1] = y[0:d - 1]
            mu[0:d - 1] = y[d - 1:]

            # compute now ~l and ~u
            c[1:d] = L[1:d, :] @ x
            lt = l - mu - c
            ut = u - mu - c

            # compute gradients avoiding catastrophic cancellation
            w = lnNormalProb(lt, ut)
            pl = np.exp(-0.5 * lt ** 2 - w) / np.sqrt(2 * math.pi)
            pu = np.exp(-0.5 * ut ** 2 - w) / np.sqrt(2 * math.pi)
            P = pl - pu

            # output the gradient
            dfdx = - mu[0:d - 1] + (P.T @ L[:, 0:d - 1]).T
            dfdm = mu - x + P
            grad = np.concatenate((dfdx, dfdm[:-1]), axis=0)

            # construct jacobian
            lt[np.isinf(lt)] = 0
            ut[np.isinf(ut)] = 0

            dP = - P ** 2 + lt * pl - ut * pu
            DL = np.tile(dP.reshape(d, 1), (1, d)) * L
            mx = DL - np.eye(d)
            xx = L.T @ DL
            mx = mx[:-1, :-1]
            xx = xx[:-1, :-1]
            J = np.block([[xx, mx.T],
                          [mx, np.diag(1 + dP[:-1])]])
            return (grad, J)

        return gradpsi

    def colperm(self):
        perm = np.arange(self.dim)
        L = np.zeros_like(self.cov)
        z = np.zeros_like(self.orig_mu)

        for j in perm.copy():
            pr = np.ones_like(z) * np.inf  # compute marginal prob.
            I = np.arange(j, self.dim)  # search remaining dimensions
            D = np.diag(self.cov)
            s = D[I] - np.sum(L[I, 0:j] ** 2, axis=1)
            s[s < 0] = self.eps
            s = np.sqrt(s)
            tl = (self.lb[I] - L[I, 0:j] @ z[0:j]) / s
            tu = (self.ub[I] - L[I, 0:j] @ z[0:j]) / s
            pr[I] = lnNormalProb(tl, tu)
            # find smallest marginal dimension
            k = np.argmin(pr)

            # flip dimensions k-->j
            jk = [j, k]
            kj = [k, j]
            self.cov[jk, :] = self.cov[kj, :]  # update rows of cov
            self.cov[:, jk] = self.cov[:, kj]  # update cols of cov
            L[jk, :] = L[kj, :]  # update only rows of L
            self.lb[jk] = self.lb[kj]  # update integration limits
            self.ub[jk] = self.ub[kj]  # update integration limits
            perm[jk] = perm[kj]  # keep track of permutation

            # construct L sequentially via Cholesky computation
            s = self.cov[j, j] - np.sum(L[j, 0:j] ** 2, axis=0)
            if s < -0.1:
                raise RuntimeError("Sigma is not positive semi-definite")
            elif s < 0:
                s = self.eps
            L[j, j] = np.sqrt(s)
            new_L = self.cov[j + 1:self.dim, j] - L[j + 1:self.dim, 0:j] @ L[j, 0:j].T
            L[j + 1:self.dim, j] = new_L / L[j, j]

            # find mean value, z(j), of truncated normal
            tl = (self.lb[j] - L[j, 0:j - 1] @ z[0:j - 1]) / L[j, j]
            tu = (self.ub[j] - L[j, 0:j - 1] @ z[0:j - 1]) / L[j, j]
            w = lnNormalProb(tl, tu)  # aids in computing expected value of trunc. normal
            z[j] = (np.exp(-.5 * tl ** 2 - w) - np.exp(-.5 * tu ** 2 - w)) / np.sqrt(2 * math.pi)
        return L, perm


def lnNormalProb(a, b):
    # computes ln(P(a<Z<b)) where Z~N(0,1) very accurately for any 'a', 'b'
    p = np.zeros_like(a)
    # case b>a>0
    I = a > 0
    if np.any(I):
        pa = lnPhi(a[I])
        pb = lnPhi(b[I])
        p[I] = pa + np.log1p(-np.exp(pb - pa))
    # case a<b<0
    idx = b < 0
    if np.any(idx):
        pa = lnPhi(-a[idx])  # log of lower tail
        pb = lnPhi(-b[idx])
        p[idx] = pb + np.log1p(-np.exp(pa - pb))
    # case a < 0 < b
    I = (~I) & (~idx)
    if np.any(I):
        pa = special.erfc(-a[I] / np.sqrt(2)) / 2  # lower tail
        pb = special.erfc(b[I] / np.sqrt(2)) / 2  # upper tail
        p[I] = np.log1p(-pa - pb)
    return p


def lnPhi(x):
    # computes logarithm of  tail of Z~N(0,1) mitigating numerical roundoff errors
    out = -0.5 * x ** 2 - np.log(2) + np.log(special.erfcx(x / np.sqrt(2)) + EPS)  # divide by zeros error -> add eps
    return out