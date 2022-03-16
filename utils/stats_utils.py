import numpy as np
from scipy.stats import norm


def calculate_likelihood(mu, var, x):
    return norm.pdf(x, loc=mu, scale=np.sqrt(var))


def calculate_log_likelihood(mu, var, x):
    res = norm.logpdf(x, loc=mu, scale=np.sqrt(var))
    # es[res > 0] = 0  # catch numerical issues
    return res
