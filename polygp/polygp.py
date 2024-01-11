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
from jaxknots.knots import fixknots
from pypolychord import PolyChordSettings, run_polychord
from pypolychord.priors import GaussianPrior, SortedUniformPrior, UniformPrior
from scipy.special import logsumexp
from tinygp.helpers import JAXArray
from tinygp.kernels import ExpSquared, RationalQuadratic

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
# plt.style.use("physrev.mplstyle")
# from tinygp.kernels import DotProduct, ExpSquared, RationalQuadratic


class SpectralMixtureKernel(tinygp.kernels.Kernel):
    def __init__(self, weight, scale, freq, data_scale=1.0):
        self.weight = jnp.atleast_1d(weight)
        self.scale = jnp.atleast_1d(scale)
        self.freq = jnp.atleast_1d(freq)
        self.data_scale = data_scale

    def evaluate(self, X1, X2):
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))[..., None]

        return (
            self.data_scale
            / (len(self.freq) + 1)
            * jnp.sum(
                self.weight
                * jnp.cos(2 * jnp.pi * self.freq * tau)
                * jnp.prod(jnp.exp(-2 * jnp.pi**2 * tau**2 * self.scale), axis=0)
            )
        )


class SparseSpectralKernel(tinygp.kernels.Kernel):
    def __init__(self, freq, weight):
        self.freq = jnp.atleast_1d(freq)
        self.weight = weight.squeeze()

    def evaluate(self, X1, X2):
        # tau = jnp.atleast_1d(jnp.abs(X1 - X2))[..., None]
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))

        return (
            1
            * self.weight
            / (len(self.freq) + 1)
            * jnp.sum(jnp.cos(2 * jnp.pi * self.freq * tau), axis=-1)
        )


class ExpSquaredNoise(tinygp.kernels.Kernel):
    def __init__(self, scale, weight):
        self.scale = scale.squeeze()
        self.weight = weight.squeeze()

    def evaluate(self, X1, X2):
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))
        return (
            self.weight * jnp.exp(-2 * jnp.pi**2 * tau**2 * self.scale)
        ).squeeze()


# class GibbsKernel(tinygp.kernels.Kernel):
#     def __init___(self, weight, scale, freq, data_scale=1.0):
#         self.weight = jnp.atleast_1d(weight)
#         self.scale = jnp.atleast_1d(scale)
#         self.freq = jnp.atleast_1d(freq)
#     def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:


class MeanPrior(object):
    def __init__(self) -> None:
        self.power_prior = UniformPrior(-2, 2)
        self.y_prior = UniformPrior(-2, 2)


class NonStationarySpectralMixtureKernel(tinygp.kernels.Kernel):
    def __init__(self, weight, scale, scale_mix, freq, freq_mix, rho):
        self.weight = jnp.atleast_1d(weight)
        self.scale = jnp.atleast_1d(scale)
        self.freq = jnp.atleast_1d(freq)
        self.freq_mix = jnp.atleast_1d(freq_mix)
        self.scale_mix = jnp.atleast_1d(scale_mix)
        self.rho = jnp.atleast_1d(rho)

    def evaluate(self, X1, X2):
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))[..., None]
        # phi = (cos 2pi mu x + cos 2pi mu' x, sin 2pi mu x + sin 2pi mu' x)
        ExpSquared()

        def phi(X):
            return jnp.asarray(
                [
                    jnp.cos(2 * jnp.pi * self.freq * X[..., None])
                    + jnp.cos(2 * jnp.pi * self.freq_mix * X[..., None]),
                    jnp.sin(2 * jnp.pi * self.freq * X[..., None])
                    + jnp.sin(2 * jnp.pi * self.freq_mix * X[..., None]),
                ]
            )

        phiTphi = jnp.sum(phi(X1) * phi(X2), axis=0)
        # Sigma = jnp.asarray(
        #     [

        #     ]
        # )
        return jnp.sum(
            self.weight
            * phiTphi
            * jnp.prod(jnp.exp(-2 * jnp.pi**2 * tau**2 * self.scale), axis=0)
        )


class SpectralMixtureProcess(object):
    mean_functions = [constant, linear, power, exponential, logarithm]

    def __init__(self, *args, **kwargs):
        self.kernel = kwargs.pop("kernel", SpectralMixtureKernel)
        self.mean_function = kwargs.pop("mean_function", power_shift)
        # self.mean_function = kwargs.pop("mean_function", null)

        self.file_root = kwargs.pop("file_root", "samples")
        self.base_dir = kwargs.pop("base_dir", "chains")

        self.kernel_n_max = kwargs.pop("kernel_n_max", 1)
        self.kernel_n_prior = UniformPrior(0, self.kernel_n_max + 1)
        # self.kernel_n_prior = ExponentialPrior(2)

        # self.mean_n = kwargs.pop("mean_n", 4)

        self.n_mean = len(self.mean_functions)
        self.n_mean_prior = UniformPrior(0, self.n_mean)

        self.num_mean_components = 3

        self.X = kwargs.pop("X")
        self.Y = kwargs.pop("Y")
        self.Y_err = kwargs.pop("Y_err", np.zeros(len(self.Y)))
        self.fundamental_freq = 1 / (self.X.max() - self.X.min())
        self.sampling_freq = len(self.X) * self.fundamental_freq
        self.scale_mean = self.X.max() - self.X.min()

        # self.scale_prior = kwargs.pop(
        #     "scale_prior",
        #     UniformPrior(-20, np.log(self.fundamental_freq)),
        # )
        self.scale_prior = GaussianPrior(0, 2)
        self.noise_prior = kwargs.pop("noise_prior", GaussianPrior(0, 2))
        self.weight_prior = kwargs.pop("weight_prior", GaussianPrior(0, 2))
        # self.weight_prior = ExponentialPrior(1)
        self.freq_prior = SortedUniformPrior(
            self.fundamental_freq, self.sampling_freq * 0.5
        )
        # self.mean_prior = kwargs.pop("mean_prior", UniformPrior(-5, 5))
        self.mean_prior = kwargs.pop("mean_prior", GaussianPrior(0, 2))
        self.init_params, self.fold_function = self.initialize(self.kernel_n_max)
        self.ndims = self.init_params.shape[0]

        # self.plot_dir = kwargs.pop("plot_dir", None)
        self.plot_dir = None
        self.posterior = None

    # @partial(jit, static_argnums=(0,3))
    def process(self, kernel_params, diag, mean):
        process = tinygp.GaussianProcess(
            self.kernel(*kernel_params),
            self.X,
            mean=partial(self.mean_function, mean),
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
    def condition(self, x_new, kernel_params, diag, mean, mean_choice):
        gp = self.process(kernel_params, diag, mean, mean_choice).condition(
            self.Y, x_new, diag=diag[3]
        )[1]
        return gp.sample(jax.random.PRNGKey(0), shape=(100,))

    @partial(jit, static_argnums=(0, 6))
    def predict_logprob(self, x_new, y_new, kernel_params, diag, mean):
        gp = self.process(kernel_params, diag, mean).condition(
            self.Y, x_new, diag=diag
        )[1]
        return gp.log_probability(y_new)

    def flat_to_tree(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))
        # sample = self.fold_function(theta)
        # print(theta)
        # kernel_params = [sample["weight"],np.exp(sample["scale"]), np.concatenate([sample["freq"],[self.sampling_freq]])]
        # kernel_params = [jnp.exp(sample["weight"]),
        #                  jnp.exp(sample["scale"]), sample["freq"]]
        # kernel_params = [jnp.exp(sample["weight"]),
        #                  jnp.ones_like(sample["weight"])*1e-5, sample["freq"]]
        kernel_choice = sample["kernel_choice"].astype(int)
        # kernel_choice = -1
        kernel_params = [
            jnp.exp(sample["weight"])[:kernel_choice],
            jnp.exp(sample["scale"])[:kernel_choice],
            sample["freq"][:kernel_choice],
        ]
        # kernel_params = [
        #     jnp.exp(sample["weight"][:kernel_choice]),
        #     jnp.exp(sample["scale"][:kernel_choice]),
        #     sample["freq"][:kernel_choice],
        # ]
        diag = jnp.exp(sample["diag"])
        mean = sample["mean"]
        # mean_choice = self.mean_functions[sample["mean_choice"].astype(int)]
        # mean_choice=sample["mean_choice"].astype(int)
        # mean = 0

        return kernel_params, diag, mean

    def prior(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))
        # sample = self.fold_function(theta)
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
        params["mean"] = np.random.rand(self.num_mean_components)
        self.param_names, _ = self.create_param_names(params)
        return ravel_pytree(params)

    def create_param_names(self, params):
        name_map = {
            "weight": r"\ln w_{}",
            "scale": r"\ln \sigma_{}",
            "freq": r"\mu_{}",
            "freq_mix": r"\mu'_{}",
            "diag": r"\delta_{}",
            "mean_choice": r"\alpha_m",
            "kernel_choice": r"\alpha_k",
            "mean": r"\phi_{}",
        }
        temp_name_dict = {}
        for k, v in params.items():
            temp_name_dict[k] = [
                name_map[k].format(str(i)) for i, x in enumerate(np.atleast_1d(v))
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
            os.path.join(settings.base_dir, settings.file_root) + ".paramnames",
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

    def sample_predict(self, nsamps, x_plot):
        samps = self.posterior.sample(nsamps).to_numpy()[..., :-3]
        y_samples = []
        y_std_samples = []
        data = []
        for i, s in enumerate(samps):
            y = self.predict(x_plot, *self.flat_to_tree(s))
            if not np.isnan(y[0].mean()):
                y_samples.append(y[0])
                y_std_samples.append(y[1])
            data.append(
                [
                    np.random.normal(m[0], jnp.sqrt(m[1]), size=100)
                    for m in zip(y[0], y[1])
                ]
            )
        data = np.asarray(data)
        data = np.moveaxis(data, [0, 2], [-1, -2])
        data = data.reshape(data.shape[0], -1)
        return data

    def plot_observed(
        self,
        xtest=None,
        ytest=None,
        xtrans=lambda x: x,
        ytrans=lambda y: y,
        filename=None,
        y_true=None,
        y_std=0.0,
    ):
        if not self.plot_dir:
            self.make_plot_dir()

        samps = self.posterior.sample(100).to_numpy()[..., :-3]
        f, a = plt.subplots(figsize=[3, 2])
        # a.scatter(
        #     xtrans(self.X),
        #     ytrans(self.Y),
        #     color="black",
        #     s=1,
        #     zorder=8,
        #     label="Training",
        # )
        a.errorbar(
            xtrans(self.X),
            ytrans(self.Y),
            yerr=self.Y_err * y_std,
            fmt="o",
            color="black",
            markersize=1,
            zorder=8,
            label="Training",
            capsize=0,
            elinewidth=0.5,
        )
        # 3
        if xtest is not None:
            a.scatter(
                xtrans(xtest),
                ytrans(ytest),
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

        y_samples = []
        y_std_samples = []
        diags = []
        # data = []
        H0 = []
        for i, s in enumerate(samps):
            # data.append(self.condition(x_plot, *self.flat_to_tree(s)))
            y = self.predict(x_plot, *self.flat_to_tree(s))
            h0 = self.predict(np.asarray([0.0]), *self.flat_to_tree(s))
            H0.append(h0[0])
            if not np.isnan(y[0].mean()):
                y_samples.append(y[0])
                y_std_samples.append(y[1])
                diags.append(self.flat_to_tree(s)[1])

        # data = np.asarray(data)
        # data = np.moveaxis(data, 2, 0).reshape(data.shape[2], -1)
        # y_mean = data.mean(axis=-1)
        y_mean = np.asarray(y_samples).mean(axis=0)
        y_std = np.asarray(y_samples).std(axis=0)
        # plot_var = data.std(axis=-1)
        plot_var = np.asarray(y_std + y_std_samples).mean(axis=0)
        a.plot(
            xtrans(x_plot),
            ytrans(y_mean),
            color="C0",
            label="Posterior prediction",
        )

        a.fill_between(
            xtrans(x_plot),
            ytrans(y_mean - 2 * plot_var),
            ytrans(y_mean + 2 * plot_var),
            color="C0",
            alpha=0.3,
        )

        a.fill_between(
            xtrans(x_plot),
            ytrans(y_mean - plot_var),
            ytrans(y_mean + plot_var),
            color="C0",
            alpha=0.5,
        )

        if y_true is not None:
            a.plot(
                xtrans(x_plot),
                y_true(x_plot),
                color="C1",
                label="Truth",
                alpha=0.8,
            )
        a.set_xlim(xtrans(x_plot.min()), xtrans(x_plot.max()))
        a.set_ylabel(r"$y$")
        a.set_xlabel(r"$x$")
        a.text(
            0.95,
            0.05,
            r"Ours",
            transform=a.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
        )
        a.text(
            0.05,
            0.9,
            r"\textsc{%s}" % filename,
            transform=a.transAxes,
            ha="left",
            va="bottom",
            fontsize=6,
        )
        f.tight_layout(pad=0.1)
        f.savefig(os.path.join(self.plot_dir, "plot_observed_%s.pdf" % filename))
        f.savefig(os.path.join(self.plot_dir, "plot_observed_%s.png" % filename))

    def nlpd(self, xtest, ytest, ytrans, y_std):
        self.read_samples()
        from scipy.stats import norm

        samps = self.posterior.sample(200).to_numpy()[..., :-3]

        logps = []
        y_samples = []
        y_std_samples = []
        # logps_2=[]
        for i, s in enumerate(samps):
            prob = self.predict_logprob(xtest, ytest, *self.flat_to_tree(s))
            logps.append(prob)
            y = self.predict(xtest, *self.flat_to_tree(s))
            y_samples.append(y[0])
            y_std_samples.append(y[1])
            # logps_2.append(
            #     norm(
            #         loc=ytrans(y[0]), scale=jnp.sqrt(y[1] * y_std**2)
            #     ).logpdf(ytrans(ytest))
            # )

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
        if not self.plot_dir:
            self.make_plot_dir()
        idx_k = [i for i, j in self.posterior.columns if "alpha_k" in j]
        # idx_m = [i for i, j in self.posterior.columns if "alpha_m" in j]
        import matplotlib.ticker as ticker

        f, a = plt.subplots(ncols=1, figsize=[3, 2], sharey=True)

        x = self.posterior[idx_k].compress(100)
        heights, bins = np.histogram(x.astype(int), np.arange(0, self.kernel_n_max + 2))

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
        a.set_xlim([-0.5, self.kernel_n_max + 0.5])
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

    def plot_corners(self, true=np.empty(0)):
        if not self.plot_dir:
            self.make_plot_dir()

        kernel_params = ["w", "sigma", "mu", "delta", "phi"][::-1]
        for kp in kernel_params:
            idx = [i for i, j in self.posterior.columns if kp in j]
            if kp == "alpha":
                x = self.posterior[idx].compress(1000)
                import matplotlib.ticker as ticker

                heights, bins = np.histogram(
                    x.astype(int), np.arange(0, self.kernel_n_max + 1)
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
                a.set_xlim([-0.5, self.kernel_n_max + 0.5])
                a.set_ylim([0, 1])
                f.tight_layout(pad=0.1)
                f.savefig(os.path.join(self.plot_dir, "alpha_hist.pdf"))
            f, a = anesthetic.make_2d_axes(idx, upper=True, figsize=[6.5, 6.5])

            post = self.posterior[idx]
            post = post.compress(500)
            # post = self.posterior[idx]
            a = post.plot_2d(
                a,
                kinds=dict(diagonal="kde_1d", lower="kde_2d", upper="scatter_2d"),
                label="Posterior",
                cmap=plt.cm.magma_r,
                upper_kwargs=dict(color="black"),
                lower_kwargs=dict(
                    cmap=plt.cm.magma_r,
                    # levels=[0.997, 0.954, 0.683, 0.32, 0.1],
                ),
            )

            if true.any():
                a.axlines(
                    idx,
                    true[np.asarray(idx).astype(np.int)].squeeze(),
                    color="black",
                    linestyle=(1, (1, 1)),
                    linewidth=0.5,
                )
            for i, r in a.items():
                for j, c in r.items():
                    if c:
                        c.tick_params(axis="x", labelsize=6)
                        c.tick_params(axis="y", labelsize=6)
            f.tight_layout(pad=0.1)
            f.savefig(os.path.join(self.plot_dir, "corner_{}.pdf").format(kp))
            f.savefig(os.path.join(self.plot_dir, "corner_{}.png").format(kp))


class NonStationarySpectralMixtureProcess(SpectralMixtureProcess):
    # def __init__(self, *args, **kwargs):
    #     # self.kernel = NonStationarySpectralMixtureKernel
    #     super().__init__(*args,kernel = NonStationarySpectralMixtureKernel, **kwargs)
    def __init__(self, *args, **kwargs):
        self.kernel = kwargs.pop("kernel", NonStationarySpectralMixtureKernel)

        self.file_root = kwargs.pop("file_root", "samples")
        self.base_dir = kwargs.pop("base_dir", "chains")

        self.kernel_n_max = kwargs.pop("kernel_n_max", 1)
        # self.kernel_n_prior = UniformPrior(0, self.kernel_n_max + 1)
        self.kernel_n_prior = ExponentialPrior(2)

        # self.mean_n = kwargs.pop("mean_n", 4)

        # self.n_mean = len(self.mean_functions)
        # self.n_mean_prior = UniformPrior(0, self.n_mean)

        # self.num_mean_components = 4

        self.X = kwargs.pop("X")
        self.Y = kwargs.pop("Y")

        self.fundamental_freq = 1 / (self.X.max() - self.X.min())
        self.sampling_freq = len(self.X) * self.fundamental_freq
        self.scale_mean = self.X.max() - self.X.min()

        # self.scale_prior = kwargs.pop(
        #     "scale_prior",
        #     UniformPrior(-20, np.log(self.fundamental_freq)),
        # )
        self.scale_prior = GaussianPrior(0, 2)
        self.noise_prior = kwargs.pop("noise_prior", GaussianPrior(0, 2))
        self.weight_prior = kwargs.pop("weight_prior", GaussianPrior(0, 2))
        # self.weight_prior = ExponentialPrior(1)
        self.scale_prior = kwargs.pop(
            "scale_prior",
            UniformPrior(-20, np.log(self.fundamental_freq)),
        )
        self.freq_prior = SortedUniformPrior(
            self.fundamental_freq, self.sampling_freq * 0.5
        )
        self.freq_prior_mix = UniformPrior(
            self.fundamental_freq, self.sampling_freq * 0.5
        )
        # self.mean_prior = kwargs.pop("mean_prior", UniformPrior(-50, 50))
        self.init_params, self.fold_function = self.initialize(self.kernel_n_max)
        self.ndims = self.init_params.shape[0]

        # self.plot_dir = kwargs.pop("plot_dir", None)
        self.plot_dir = None
        self.posterior = None

    # @partial(jit, static_argnums=(0,3))
    def process(self, kernel_params, diag):
        process = tinygp.GaussianProcess(
            self.kernel(
                *kernel_params
            ),  # + diag[2] * (1 + ExpSquared(scale=2)), #RationalQuadratic(scale=diag[0],alpha=diag[1]),
            self.X,
            diag=diag,
        )
        return process

    @partial(jit, static_argnums=(0,))
    def logl(self, kernel_params, diag):
        return self.process(kernel_params, diag).log_probability(self.Y)

    @partial(jit, static_argnums=(0,))
    def predict(self, x_new, kernel_params, diag):
        gp = self.process(kernel_params, diag).condition(self.Y, x_new, diag=diag)[1]
        return gp.loc, gp.variance

    @partial(jit, static_argnums=(0,))
    def condition(self, x_new, kernel_params, diag):
        gp = self.process(kernel_params, diag).condition(self.Y, x_new, diag=diag)[1]
        return gp.sample(jax.random.PRNGKey(0), shape=(100,))

    @partial(jit, static_argnums=(0))
    def predict_logprob(self, x_new, y_new, kernel_params, diag):
        gp = self.process(kernel_params, diag).condition(self.Y, x_new, diag=diag)[1]
        return gp.log_probability(y_new)

    def flat_to_tree(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))
        # kernel_choice = sample["kernel_choice"].astype(int)
        kernel_choice = self.kernel_n_max
        kernel_params = [
            jnp.exp(sample["weight"][:kernel_choice]),
            jnp.exp(sample["scale"][:kernel_choice]),
            sample["freq"][:kernel_choice],
            sample["freq_mix"][:kernel_choice],
        ]
        diag = jnp.exp(sample["diag"])
        return kernel_params, diag

    def prior(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))
        # sample = self.fold_function(theta)
        sample["weight"] = self.weight_prior(sample["weight"])
        sample["scale"] = self.scale_prior(sample["scale"])
        sample["freq"] = self.freq_prior(sample["freq"])
        sample["freq_mix"] = self.freq_prior_mix(sample["freq_mix"])
        sample["diag"] = self.noise_prior(sample["diag"])
        # sample["kernel_choice"] = self.kernel_n_prior(sample["kernel_choice"])
        t, _ = ravel_pytree(sample)
        return t

    def initialize(self, n):
        params = {}
        params["weight"] = np.random.rand(n)
        params["scale"] = np.random.rand(n)
        params["freq"] = np.random.rand(n)
        params["freq_mix"] = np.random.rand(n)
        params["diag"] = np.random.randn()
        # params["mean_choice"] = np.random.randn()
        # params["kernel_choice"] = np.random.rand()
        # params["mean"] = np.random.rand(self.num_mean_components)
        self.param_names, _ = self.create_param_names(params)
        return ravel_pytree(params)


class StaticSpectralMixtureProcess(SpectralMixtureProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, mean_function=null, num_mean_components=0, **kwargs)

    def flat_to_tree(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))
        # kernel_choice = sample["kernel_choice"].astype(int)
        kernel_choice = self.kernel_n_max
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
        # sample = self.fold_function(theta)
        sample["weight"] = self.weight_prior(sample["weight"])
        sample["scale"] = self.scale_prior(sample["scale"])
        sample["freq"] = self.freq_prior(sample["freq"])
        sample["diag"] = self.noise_prior(sample["diag"])
        # sample["kernel_choice"] = self.kernel_n_prior(sample["kernel_choice"])
        t, _ = ravel_pytree(sample)
        return t

    def initialize(self, n):
        params = {}
        params["weight"] = np.random.rand(n)
        params["scale"] = np.random.rand(n)
        params["freq"] = np.random.rand(n)
        # params["freq_mix"] = np.random.rand(n)
        params["diag"] = np.random.randn()
        # params["mean_choice"] = np.random.randn()
        # params["kernel_choice"] = np.random.rand()
        # params["mean"] = np.random.rand(self.num_mean_components)
        self.param_names, _ = self.create_param_names(params)
        return ravel_pytree(params)


class SparseSpectralMixtureProcess(SpectralMixtureProcess):
    # def __init__(self, *args, **kwargs):
    #     # self.kernel = NonStationarySpectralMixtureKernel
    #     super().__init__(*args,kernel = NonStationarySpectralMixtureKernel, **kwargs)
    def __init__(self, *args, **kwargs):
        self.kernel = kwargs.pop("kernel", SparseSpectralKernel)

        self.file_root = kwargs.pop("file_root", "samples")
        self.base_dir = kwargs.pop("base_dir", "chains")

        self.kernel_n_max = kwargs.pop("kernel_n_max", 1)
        # self.kernel_n_prior = UniformPrior(0, self.kernel_n_max + 1)
        self.kernel_n_prior = ExponentialPrior(2)

        self.mean = fixknots(n=3)

        # self.mean_n = kwargs.pop("mean_n", 4)

        # self.n_mean = len(self.mean_functions)
        # self.n_mean_prior = UniformPrior(0, self.n_mean)

        self.num_mean_components = 4

        self.X = kwargs.pop("X")
        self.Y = kwargs.pop("Y")

        self.fundamental_freq = 1 / (self.X.max() - self.X.min())
        self.sampling_freq = len(self.X) * self.fundamental_freq
        self.scale_mean = self.X.max() - self.X.min()

        # self.scale_prior = kwargs.pop(
        #     "scale_prior",
        #     UniformPrior(-20, np.log(self.fundamental_freq)),
        # )
        self.scale_prior = GaussianPrior(0, 2)
        self.noise_prior = kwargs.pop("noise_prior", GaussianPrior(0, 2))
        self.weight_prior = kwargs.pop("weight_prior", GaussianPrior(0, 2))
        # self.weight_prior = ExponentialPrior(1)
        self.scale_prior = kwargs.pop(
            "scale_prior",
            UniformPrior(-20, np.log(self.fundamental_freq)),
        )
        self.freq_prior = SortedUniformPrior(
            self.fundamental_freq, self.sampling_freq * 0.5
        )
        self.freq_prior_mix = UniformPrior(
            self.fundamental_freq, self.sampling_freq * 0.5
        )
        self.mean_prior = kwargs.pop("mean_prior", UniformPrior(-5, 5))
        self.init_params, self.fold_function = self.initialize(self.kernel_n_max)
        self.ndims = self.init_params.shape[0]

        # self.plot_dir = kwargs.pop("plot_dir", None)
        self.plot_dir = None
        self.posterior = None

    # @partial(jit, static_argnums=(0,3))
    def process(self, kernel_params, diag, mean, scale):
        process = tinygp.GaussianProcess(
            self.kernel(*kernel_params),
            # + ExpSquaredNoise(
            #     scale=scale, weight=kernel_params[-1]
            # ),  # *  ExpSquared(scale=scale),  # + diag[2] * (1 + ExpSquared(scale=2)), #RationalQuadratic(scale=diag[0],alpha=diag[1]),
            self.X,
            diag=diag,
            mean=partial(self.mean, mean),
        )
        return process

    @partial(jit, static_argnums=(0,))
    def logl(self, kernel_params, diag, mean, scale):
        return self.process(kernel_params, diag, mean, scale).log_probability(self.Y)

    @partial(jit, static_argnums=(0,))
    def predict(self, x_new, kernel_params, diag, mean, scale):
        gp = self.process(kernel_params, diag, mean, scale).condition(
            self.Y, x_new, diag=diag
        )[1]
        return gp.loc, gp.variance

    @partial(jit, static_argnums=(0,))
    def condition(self, x_new, kernel_params, diag, mean, scale):
        gp = self.process(kernel_params, diag, mean, scale).condition(
            self.Y, x_new, diag=diag
        )[1]
        return gp.sample(jax.random.PRNGKey(0), shape=(100,))

    @partial(jit, static_argnums=(0))
    def predict_logprob(self, x_new, y_new, kernel_params, diag, mean):
        gp = self.process(kernel_params, diag, mean).condition(
            self.Y, x_new, diag=diag
        )[1]
        return gp.log_probability(y_new)

    def flat_to_tree(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))
        # kernel_choice = sample["kernel_choice"].astype(int)
        # kernel_choice = self.kernel_n_max
        kernel_params = [sample["freq"], jnp.exp(sample["weight"])]
        scale = jnp.exp(sample["weight"])
        # kernel_params = [
        #     jnp.exp(sample["weight"][:kernel_choice]),
        #     jnp.exp(sample["scale"][:kernel_choice]),
        #     sample["freq"][:kernel_choice],
        #     sample["freq_mix"][:kernel_choice],
        # ]
        diag = jnp.exp(sample["diag"])
        mean = sample["mean"]
        return kernel_params, diag, mean, scale

    def prior(self, theta):
        sample = self.fold_function(theta.astype(jnp.float32))
        # sample = self.fold_function(theta)
        sample["weight"] = self.weight_prior(sample["weight"])
        sample["scale"] = self.scale_prior(sample["scale"])
        sample["freq"] = self.freq_prior(sample["freq"])
        # sample["freq_mix"] = self.freq_prior_mix(sample["freq_mix"])
        sample["diag"] = self.noise_prior(sample["diag"])
        # sample["mean"] = self.mean_prior(sample["mean"])
        sample["mean"] = sample["mean"]
        # sample["kernel_choice"] = self.kernel_n_prior(sample["kernel_choice"])
        t, _ = ravel_pytree(sample)
        return t

    def initialize(self, n):
        params = {}
        params["weight"] = np.random.randn()
        params["scale"] = np.random.randn()
        params["freq"] = np.random.rand(n)
        # params["freq_mix"] = np.random.rand(n)
        params["diag"] = np.random.randn()
        # params["mean_choice"] = np.random.randn()
        # params["kernel_choice"] = np.random.rand()
        params["mean"] = np.random.rand(self.num_mean_components)
        self.param_names, _ = self.create_param_names(params)
        return ravel_pytree(params)
