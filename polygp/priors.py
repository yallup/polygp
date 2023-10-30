"""Extra priors defined above PolyChord if needed"""

from scipy.special import erfinv
from pypolychord.priors import (
    SortedUniformPrior,
    UniformPrior,
    GaussianPrior,
    forced_indentifiability_transform,
)

from scipy import stats as stats
import numpy as np




class BetaPrior(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, cube):
        icdf = stats.beta.ppf(cube, self.alpha, self.beta)
        return icdf


class SortedGaussianPrior(GaussianPrior):
    def __call__(self, x):
        t = forced_indentifiability_transform(x)
        return super(SortedGaussianPrior, self).__call__(t)


class ExponentialPrior:
    def __init__(self, lam):
        self.lam = lam

    def __call__(self, x):
        return -np.log(1 - x + 1e-30) / self.lam


class SortedExponentialPrior(ExponentialPrior):
    def __call__(self, x):
        t = forced_indentifiability_transform(x)
        return super(SortedExponentialPrior, self).__call__(t)


class LogGaussianPrior:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(self.mu + self.sigma * np.sqrt(2) * erfinv(2 * x - 1))
