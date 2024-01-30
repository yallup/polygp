#!/usr/bin/env python3
import os

# from jax import lax
import warnings
from functools import partial

import anesthetic
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tinygp
from jax import jit
from jax.config import config
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten

# from jaxknots.knots import fixknots
from pypolychord import PolyChordSettings, run_polychord
from pypolychord.priors import GaussianPrior, SortedUniformPrior, UniformPrior
from scipy.special import logsumexp
from tinygp.helpers import JAXArray
from tinygp.kernels import (
    DotProduct,
    ExpSquared,
    Polynomial,
    RationalQuadratic,
)

from polygp.means import (
    constant,
    exponential,
    linear,
    logarithm,
    null,
    power,
    power_shift,
)
from polygp.priors import ExponentialPrior

warnings.filterwarnings("ignore")
# config.update("jax_enable_x64", True)


class SpectralMixtureKernel(tinygp.kernels.Kernel):
    def __init__(self, weight, scale, freq):
        self.weight = jnp.atleast_1d(weight)
        self.scale = jnp.atleast_1d(scale)
        self.freq = jnp.atleast_1d(freq)

    def evaluate(self, X1, X2):
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))[..., None]

        return jnp.sum(
            self.weight
            * jnp.prod(
                jnp.cos(2 * jnp.pi * self.freq * tau)
                * jnp.exp(-2 * jnp.pi**2 * tau**2 * self.scale),
                axis=0,
            )
        )


class SpectralMixtureProcess(object):

    """
    A Spectral Mixture Process with an adaptive mean function and adaptive kernel.

    Args:
        n (int, optional): Number of components in the kernel. Defaults to 3.
        **kwargs: Keyword arguments passed to the kernel and mean function.

    Keyword Args:
        kernel (tinygp.kernels.Kernel, optional): Kernel function. Defaults to SpectralMixtureKernel.
        mean (callable, optional): Mean function. Defaults to a power law.
        file_root (str, optional): File root for PolyChord output. Defaults to "samples".
        base_dir (str, optional): Base directory for PolyChord output. Defaults to "chains".
        mean_n (int, optional): Number of components in the mean function. Defaults to 3 (corresponds to power law).
        X (array_like): Training data x points.
        Y (array_like): Training data y points.
        Y_err (array_like, optional): Training data y errors. Defaults to np.zeros(len(Y)).
        scale_prior (pypolychord.priors.Prior, optional): Prior on the LOG of the scale parameter. Defaults to Gaussian(0,2).
        noise_prior (pypolychord.priors.Prior, optional): Prior on the LOG of the noise parameter. Defaults to Gaussian(0,2).
        weight_prior (pypolychord.priors.Prior, optional): Prior on the LOG weight parameter. Defaults to Gaussian(0,2).
        freq_prior (pypolychord.priors.Prior, optional): Prior on the frequency parameter. Defaults to SortedUniformPrior(fundamental_freq, 0.5 *sampling_freq).
        mean_prior (pypolychord.priors.Prior, optional): Prior on the components of thmean mean function. Defaults to Gaussian(0,1).

    """

    def __init__(self, n=3, **kwargs):
        self.kernel = kwargs.pop("kernel", SpectralMixtureKernel)
        self.mean = kwargs.pop("mean", power_shift)
        # self.mean_function = kwargs.pop("mean_function", null)
        # self.mean_function = kwargs.pop("mean_function", linear)
        self.file_root = kwargs.pop("file_root", "samples")
        self.base_dir = kwargs.pop("base_dir", "chains")

        self.kernel_n = n
        self.kernel_n_prior = UniformPrior(0, self.kernel_n + 1)
        # self.kernel_n_prior = ExponentialPrior(1/self.kernel_n_max)

        self.mean_n = kwargs.pop("mean_n", 3)
        self.n_mean_prior = UniformPrior(0, self.mean_n + 1)

        self.X = kwargs.pop("X")
        self.Y = kwargs.pop("Y")
        self.Y_err = kwargs.pop("Y_err", np.zeros(len(self.Y)))
        self.xtrans = None
        self.ytrans = None
        self.whiten()

        self.fundamental_freq = 1 / (self.X.max() - self.X.min())
        self.sampling_freq = len(self.X) * self.fundamental_freq
        self.scale_mean = self.X.max() - self.X.min()

        # self.scale_prior = kwargs.pop("scale_prior", UniformPrior(-3, 3))
        self.scale_prior = kwargs.pop("scale_prior", GaussianPrior(0, 2))
        self.noise_prior = kwargs.pop("noise_prior", GaussianPrior(0, 2))
        self.weight_prior = kwargs.pop("weight_prior", GaussianPrior(0, 2))
        # self.weight_prior = ExponentialPrior(1)
        self.freq_prior = SortedUniformPrior(
            self.fundamental_freq, self.sampling_freq * 0.5
        )
        # self.freq_prior = SortedUniformPrior(0, self.sampling_freq)
        self.mean_prior = kwargs.pop("mean_prior", GaussianPrior(0, 1))
        # self.mean_prior = kwargs.pop("mean_prior", UniformPrior(-5, 5))

        self.init_params, self.fold_function = self.initialize(
            self.kernel_n
        )
        self.ndims = self.init_params.shape[0]

        self.plot_dir = None
        self.posterior = None

    def whiten(self):
        if not self.ytrans:
            y_std = self.Y.std()
            y_mean = self.Y.mean()
            self.ytrans = lambda y: y * y_std + y_mean
            self.Y = (self.Y - y_mean) / y_std
            self.Y_err /= y_std
        if not self.xtrans:
            x_min = self.X.min()
            x_max = self.X.max()
            self.xtrans = lambda x: (x - x_min) / (x_max - x_min)
            self.X = self.X * (x_max - x_min) + x_min

    # @partial(jit, static_argnums=(0,3))
    def process(self, kernel_params, diag, mean):
        process = tinygp.GaussianProcess(
            self.kernel(*kernel_params),
            self.X,
            mean=partial(self.mean, mean),
            diag=diag + self.Y_err,
        )
        return process

    @partial(jit, static_argnums=(0,))
    def logl(self, kernel_params, diag, mean):
        return self.process(kernel_params, diag, mean).log_probability(self.Y)

    @partial(jit, static_argnums=(0,))
    def predict(self, x_new, kernel_params, diag, mean):
        gp = self.process(kernel_params, diag, mean).condition(
            self.Y, x_new, diag=diag
        )[1]
        return gp.loc, gp.variance

    @partial(jit, static_argnums=(0, 5))
    def condition(self, x_new, kernel_params, diag, mean, n_samps=30):
        gp = self.process(kernel_params, diag, mean).condition(
            self.Y, x_new, diag=diag
        )[1]
        return gp.sample(jax.random.PRNGKey(0), shape=(n_samps,))

    @partial(jit, static_argnums=(0, 6))
    def predict_logprob(self, x_new, y_new, kernel_params, diag, mean):
        gp = self.process(kernel_params, diag, mean).condition(
            self.Y, x_new, diag=diag
        )[1]
        return gp.log_probability(y_new)

    def flat_to_tree(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))

        kernel_choice = sample["kernel_choice"].astype(int)
        mean_choice = sample["mean_choice"].astype(int)
        kernel_params = [
            sample["weight"][:kernel_choice],
            jnp.exp(sample["scale"])[:kernel_choice],
            sample["freq"][:kernel_choice],
        ]
        diag = jnp.exp(sample["diag"])
        mean = sample["mean"]
        mean = np.array(mean)
        mean[mean_choice:] = 0.0
        return kernel_params, diag, mean

    def prior(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))
        sample["weight"] = self.weight_prior(sample["weight"])
        sample["scale"] = self.scale_prior(sample["scale"])
        sample["freq"] = self.freq_prior(sample["freq"])
        sample["mean"] = self.mean_prior(sample["mean"])
        sample["diag"] = self.noise_prior(sample["diag"])
        sample["mean_choice"] = self.n_mean_prior(sample["mean_choice"])
        sample["kernel_choice"] = self.kernel_n_prior(sample["kernel_choice"])
        t, _ = ravel_pytree(sample)
        return t

    def initialize(self, n):
        params = {}
        params["weight"] = np.random.rand(n)
        params["scale"] = np.random.rand(n)
        params["freq"] = np.random.rand(n)
        params["diag"] = np.random.rand()
        params["mean_choice"] = np.random.randn()
        params["kernel_choice"] = np.random.rand()
        params["mean"] = np.random.rand(self.mean_n)
        self.param_names, _ = self.create_param_names(params)
        return ravel_pytree(params)

    def create_param_names(self, params):
        name_map = {
            "weight": r"\ln w_{}",
            "scale": r"\ln \sigma_{}",
            "freq": r"\mu_{}",
            "diag": r"\delta_{}",
            "kernel_choice": r"\alpha_k",
            "kernel_choice_lin": r"\alpha_k'",
            "mean": r"\phi_{}",
        }
        temp_name_dict = {}
        for k, v in params.items():
            temp_name_dict[k] = [
                name_map[k].format(str(i))
                for i, x in enumerate(np.atleast_1d(v))
            ]
        return tree_flatten(temp_name_dict)

    def train(self, **kwargs):
        settings = PolyChordSettings(self.ndims, 0)
        settings.nlive = kwargs.pop("nlive", settings.nlive)
        settings.num_repeats = kwargs.pop("fac_repeat", 2) * self.ndims
        settings.nprior = kwargs.pop("fac_prior", 10) * settings.nlive

        if kwargs:
            raise TypeError("Unexpected **kwargs: %r" % kwargs)

        settings.read_resume = False
        settings.write_resume = False
        settings.posteriors = False
        settings.equals = False
        settings.file_root = self.file_root
        settings.base_dir = self.base_dir
        settings.precision_criterion = 1e-3

        def wrapped_loglikelihood(theta):
            try:
                logl = self.logl(*self.flat_to_tree(theta))
            except RuntimeError:
                logl = settings.logzero

            if np.isnan(logl):
                logl = settings.logzero
            if np.isinf(logl):
                logl = settings.logzero
            return float(logl), []

        out = run_polychord(
            wrapped_loglikelihood, self.ndims, 0, settings, prior=self.prior
        )
        with open(
            os.path.join(settings.base_dir, settings.file_root)
            + ".paramnames",
            "w",
        ) as f:
            for i, name in enumerate(self.param_names):
                f.write("%s   %s\n" % (i, name))
        return out

    def make_plot_dir(self):
        self.plot_dir = os.path.join(self.base_dir, self.file_root) + "_plots"
        self.read_samples()
        os.makedirs(self.plot_dir, exist_ok=True)

    def read_samples(self):
        try:
            self.posterior = anesthetic.read_chains(
                os.path.join(self.base_dir, self.file_root)
            )
        except AttributeError:
            raise "No posterior chains found"

    def sample_predict(self, nsamps, x, process_samps=30):
        """
        From a trained process predict samples from the process at new x points.

        Parameters
        ----------
        nsamps : int
            Number of samples to draw from the posterior.
        x : array_like
            New x points to predict at.

        Returns
        -------
        data : array_like
            Samples from the posterior at x.

        """
        samps = self.posterior.sample(nsamps).to_numpy()[..., :-3]
        y_samples = []

        for i, s in enumerate(samps):
            # y = self.predict(x, *self.flat_to_tree(s))
            y = self.condition(x, *self.flat_to_tree(s), n_samps=process_samps)
            y_samples.append(y)
        y_samples = np.asarray(y_samples)
        # y_samples = np.moveaxis(y_samples, [0, 2], [-1, -2])
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])
        return y_samples

    def plot_observed(
        self,
        xtest=None,
        ytest=None,
        filename=None,
    ):
        """
        A convenience function to plot the observed data and the posterior

        Parameters
        ----------
        xtest : array_like
            New x points to predict at.
        ytest : array_like
            New y points to predict at.

        Returns
        -------
        f : matplotlib.figure.Figure
            Figure object
        a : matplotlib.axes.Axes
            Axes object
        """
        if not self.plot_dir:
            self.make_plot_dir()

        f, a = plt.subplots(figsize=[4, 3])

        a.errorbar(
            self.xtrans(self.X),
            self.ytrans(self.Y),
            yerr=self.ytrans(self.Y_err + self.Y) - self.ytrans(self.Y),
            fmt="o",
            color="black",
            markersize=1,
            zorder=8,
            label="Training",
            capsize=0,
            elinewidth=0.5,
        )

        if xtest is not None:
            a.scatter(
                xtest,
                ytest,
                color="grey",
                s=1,
                zorder=8,
                label="Test",
            )
        if xtest is None:
            x_plot = np.linspace(self.X.min(), self.X.max(), 200)
        else:
            x_plot = np.linspace(
                np.min([xtest.min(), self.X.min()]),
                np.max([xtest.max(), self.X.max()]),
                200,
            )

        y_samples = self.sample_predict(100, x_plot)
        y_mean = np.asarray(y_samples).mean(axis=0)
        y_std = np.asarray(y_samples).std(axis=0)

        a.plot(
            self.xtrans(x_plot),
            self.ytrans(y_mean),
            color="C0",
            label="Posterior prediction",
        )

        a.fill_between(
            self.xtrans(x_plot),
            self.ytrans(y_mean - 2 * y_std),
            self.ytrans(y_mean + 2 * y_std),
            color="C0",
            alpha=0.3,
        )

        a.fill_between(
            self.xtrans(x_plot),
            self.ytrans(y_mean - y_std),
            self.ytrans(y_mean + y_std),
            color="C0",
            alpha=0.5,
        )

        a.set_xlim(self.xtrans(x_plot.min()), self.xtrans(x_plot.max()))
        f.tight_layout(pad=0.1)
        f.savefig(
            os.path.join(self.plot_dir, "plot_observed_%s.pdf" % filename)
        )
        f.savefig(
            os.path.join(self.plot_dir, "plot_observed_%s.png" % filename)
        )
        return f, a

    def nlpd(self, xtest, ytest, y_std):
        """
        Calculate scores of model on new data

        Args:
            xtest (array_like): New x points to predict at.
            ytest (array_like): New y points to predict at.
        """
        
        self.read_samples()
        samps = self.posterior.sample(200).to_numpy()[..., :-3]
        logps = []
        y_samples = []
        y_std_samples = []
        for i, s in enumerate(samps):
            prob = self.predict_logprob(xtest, ytest, *self.flat_to_tree(s))
            logps.append(prob)
            y = self.predict(xtest, *self.flat_to_tree(s))
            y_samples.append(y[0])
            y_std_samples.append(y[1])

        y_mean = np.asarray(y_samples).mean(axis=0)
        y_std = np.asarray(y_samples).std(axis=0)
        logps = np.asarray(logps)
        # # print(logps)
        # return (
        #     logsumexp(np.asarray(logps), axis=0)
        #     - np.log(np.asarray(logps).shape[0])
        # ).mean()
        return (
            np.mean(y_mean - ytest),
            np.mean(-((y_mean - ytest) ** 2) / (y_std + y_std_samples) ** 2),
            np.mean(logps) / len(ytest),
        )

    def plot_n_components(self):
        """
        Plot a histogram of the number of components in the kernel.
        """

        if not self.plot_dir:
            self.make_plot_dir()
        idx_k = [i for i, j in self.posterior.columns if "alpha_k" in j]
        # idx_m = [i for i, j in self.posterior.columns if "alpha_m" in j]
        import matplotlib.ticker as ticker

        f, a = plt.subplots(ncols=1, figsize=[3, 2], sharey=True)

        x = self.posterior[idx_k].compress(100)
        heights, bins = np.histogram(
            x.astype(int), np.arange(0, self.kernel_n + 2)
        )

        a.hist(
            bins[:-1],
            weights=heights,
            bins=np.concatenate([[-0.5], (bins[:-1] + 0.5)]),
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        a.set_xlabel(r"$\alpha_k$")
        a.set_ylabel(r"$P(\alpha)$")
        # Remove non-integer ticks from x-axis
        a.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        a.xaxis.set_minor_locator(ticker.NullLocator())
        a.set_xlim([-0.5, self.kernel_n + 0.5])
        a.set_ylim([0, 1])

        # x = self.posterior[idx_m].compress(100)
        # heights, bins = np.histogram(
        #     x.astype(int), np.arange(0, self.n_mean + 1)
        # )

        # a[1].hist(
        #     bins[:-1],
        #     weights=heights,
        #     bins=np.concatenate([[-0.5], (bins[:-1] + 0.5)]),
        #     density=True,
        #     edgecolor="black",
        #     linewidth=0.5,
        # )
        # a[1].set_xlabel(r"$\alpha_m$")
        # # a[1].set_ylabel(r"$P(\alpha)$")
        # # Remove non-integer ticks from x-axis
        # a[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # a[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        # a[1].xaxis.set_minor_locator(ticker.NullLocator())
        # a[1].set_xlim([-0.5, self.n_mean + 0.5])
        # a[1].set_ylim([0, 1])

        f.tight_layout(pad=0.1)
        f.savefig(os.path.join(self.plot_dir, "alpha_hist.pdf"))
        f.savefig(os.path.join(self.plot_dir, "alpha_hist.png"))

    def plot_corners(self):
        """
        Plot all corner plots of
        """

        if not self.plot_dir:
            self.make_plot_dir()

        kernel_params = ["w", "sigma", "mu", "delta", "phi"][::-1]
        for kp in kernel_params:
            idx = [i for i, j in self.posterior.columns if kp in j]
            if kp == "alpha":
                x = self.posterior[idx].compress(1000)
                import matplotlib.ticker as ticker

                heights, bins = np.histogram(
                    x.astype(int), np.arange(0, self.kernel_n + 1)
                )
                f, a = plt.subplots(figsize=[3, 2])
                a.hist(
                    bins[:-1],
                    weights=heights,
                    bins=np.concatenate([[-0.5], (bins + 0.5)]),
                    density=True,
                    edgecolor="black",
                    linewidth=0.5,
                )
                a.set_xlabel(r"$\alpha$")
                a.set_ylabel(r"$P(\alpha)$")
                # Remove non-integer ticks from x-axis
                a.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
                a.xaxis.set_minor_locator(ticker.NullLocator())
                a.set_xlim([-0.5, self.kernel_n + 0.5])
                a.set_ylim([0, 1])
                f.tight_layout(pad=0.1)
                f.savefig(os.path.join(self.plot_dir, "alpha_hist.pdf"))
            f, a = anesthetic.make_2d_axes(idx, upper=True, figsize=[6.5, 6.5])

            post = self.posterior[idx]
            post = post.compress(500)
            # post = self.posterior[idx]
            a = post.plot_2d(
                a,
                kinds=dict(
                    diagonal="kde_1d", lower="kde_2d", upper="scatter_2d"
                ),
                label="Posterior",
                cmap=plt.cm.magma_r,
                upper_kwargs=dict(color="black"),
                lower_kwargs=dict(
                    cmap=plt.cm.magma_r,
                    # levels=[0.997, 0.954, 0.683, 0.32, 0.1],
                ),
            )

            # if true.any():
            #     a.axlines(
            #         idx,
            #         true[np.asarray(idx).astype(np.int)].squeeze(),
            #         color="black",
            #         linestyle=(1, (1, 1)),
            #         linewidth=0.5,
            #     )
            for i, r in a.items():
                for j, c in r.items():
                    if c:
                        c.tick_params(axis="x", labelsize=6)
                        c.tick_params(axis="y", labelsize=6)
            f.tight_layout(pad=0.1)
            f.savefig(os.path.join(self.plot_dir, "corner_{}.pdf").format(kp))
            f.savefig(os.path.join(self.plot_dir, "corner_{}.png").format(kp))


class StaticSpectralMixtureProcess(SpectralMixtureProcess):
    """
    Optimize a SM Kernel with a fixed number of components and fixed mean function (defaults to null)
    """

    def __init__(self, *args, **kwargs):
        mean = kwargs.pop("mean", null)
        mean_n = kwargs.pop("mean_n", 0)
        super().__init__(
            *args, mean=mean, mean_n=mean_n, **kwargs
        )

    def flat_to_tree(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))
        # kernel_choice = sample["kernel_choice"].astype(int)
        kernel_choice = self.kernel_n
        kernel_params = [
            jnp.exp(sample["weight"][:kernel_choice]),
            jnp.exp(sample["scale"][:kernel_choice]),
            sample["freq"][:kernel_choice],
            # sample["freq_mix"][:kernel_choice],
        ]
        diag = jnp.exp(sample["diag"])
        return kernel_params, diag, 0.0

    def prior(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))
        sample["weight"] = self.weight_prior(sample["weight"])
        sample["scale"] = self.scale_prior(sample["scale"])
        sample["freq"] = self.freq_prior(sample["freq"])
        sample["diag"] = self.noise_prior(sample["diag"])
        t, _ = ravel_pytree(sample)
        return t

    def initialize(self, n):
        params = {}
        params["weight"] = np.random.rand(n)
        params["scale"] = np.random.rand(n)
        params["freq"] = np.random.rand(n)
        params["diag"] = np.random.randn()
        self.param_names, _ = self.create_param_names(params)
        return ravel_pytree(params)


class SemiStaticSpectralMixtureProcess(SpectralMixtureProcess):
    def __init__(self, *args, **kwargs):
        mean = kwargs.pop("mean", null)
        mean_n = kwargs.pop("mean_n", 0)
        super().__init__(
            *args, mean=mean, mean_n=mean_n, **kwargs
        )

    def flat_to_tree(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))

        kernel_choice = sample["kernel_choice"].astype(int)
        kernel_params = [
            jnp.exp(sample["weight"])[:kernel_choice],
            jnp.exp(sample["scale"])[:kernel_choice],
            sample["freq"][:kernel_choice],
        ]
        diag = jnp.exp(sample["diag"])
        mean = sample["mean"]
        return kernel_params, diag, mean

    def prior(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))
        sample["weight"] = self.weight_prior(sample["weight"])
        sample["scale"] = self.scale_prior(sample["scale"])
        sample["freq"] = self.freq_prior(sample["freq"])
        sample["mean"] = self.mean_prior(sample["mean"])
        sample["diag"] = self.noise_prior(sample["diag"])
        # sample["mean_choice"] = self.n_mean_prior(sample["mean_choice"])
        sample["kernel_choice"] = self.kernel_n_prior(sample["kernel_choice"])
        t, _ = ravel_pytree(sample)
        return t

    def initialize(self, n):
        params = {}
        params["weight"] = np.random.rand(n)
        params["scale"] = np.random.rand(n)
        params["freq"] = np.random.rand(n)
        params["diag"] = np.random.rand()
        # params["mean_choice"] = np.random.randn()
        params["kernel_choice"] = np.random.rand()
        params["mean"] = np.random.rand(self.mean_n)
        self.param_names, _ = self.create_param_names(params)
        return ravel_pytree(params)
