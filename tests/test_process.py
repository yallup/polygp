import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from pytest import approx

from polygp import SpectralMixtureProcess

X = np.linspace(0, 1, 10)
Y = np.sin(X)
X_new = np.linspace(1, 1, 10)


@pytest.mark.parametrize("kernel_n_max", [1, 4])
class TestProcess:
    @pytest.fixture
    def process(self, kernel_n_max):
        return SpectralMixtureProcess(X=X, Y=Y, kernel_n_max=kernel_n_max)

    @pytest.fixture
    def theta(self, process):
        return process.init_params

    @pytest.fixture
    def prior_sample(self, theta, process):
        return process.prior(theta)

    def test_init(self, process, kernel_n_max):
        assert process.X is X
        assert process.Y is Y
        assert process.ndims == kernel_n_max * 3 + 2 + process.mean_n

    def test_prior(self, process, kernel_n_max):
        theta_samp, fold_fn = process.initialize(kernel_n_max)
        prior_samp = process.prior(theta_samp)
        assert np.all(np.isreal(prior_samp))

    @pytest.mark.parametrize("mask", [0, 1, -1])
    def test_logl(self, process, prior_sample, mask):
        folded = process.fold_function(prior_sample)
        folded["kernel_choice"] = np.array([mask])
        ravel, _ = ravel_pytree(folded)
        refolded = process.flat_to_tree(ravel)
        logl = process.logl(*refolded)
        assert not np.isnan(logl)
        assert not np.isinf(logl)
        assert logl.dtype == np.float64

    @pytest.fixture
    def tree(self, process, prior_sample):
        return process.flat_to_tree(prior_sample)

    def test_condition(self, process, tree):
        process.condition(X_new, *tree)

    def test_predict(self, process, tree):
        process.predict(X_new, *tree)
