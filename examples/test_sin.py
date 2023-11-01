import os

import numpy as np
from mpi4py import MPI

from polygp.polygp import SpectralMixtureProcess

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


plotdir = "toy"
os.makedirs(plotdir, exist_ok=True)
rng = np.random.default_rng(0)


x_min = 0.0
x_max = 1.0

x = np.linspace(x_min, 2 * x_max, 100)


def function(x):
    """A toy function with spectral components to predict"""

    a_1 = np.sin(2 * np.pi * x)
    a_2 = np.sin(2 * np.pi * x * 1.5)
    a_3 = np.sin(2 * np.pi * x * 10)
    a_4 = np.sin(2 * np.pi * x * 8.0)
    return a_1 + a_2 + a_3 + a_4


y = function(x) + rng.normal(size=(len(x))) * 0.25

cut = int(len(x) * 0.5)

x_cut = x[:cut].max()
y_mean = y[:cut].mean()
y_std = y[:cut].std()
x_min = x.min()
y_min = y.min()

# whitening transform
y = (y - y_mean) / y_std
x = (x - x.min()) / (x_cut - x.min())

x_train = x[:cut].squeeze()
y_train = y[:cut].squeeze()
x_test = x[cut:].squeeze()
y_test = y[cut:].squeeze()
y_true = function


if __name__ == "__main__":
    nlive = 50
    smp = SpectralMixtureProcess(
        X=x_train,
        Y=y_train,
        kernel_n_max=4,
        base_dir="chains",
        file_root="toy_sin",
        fac_repeat=2,
    )
    output = smp.train(nlive=nlive)

    if rank == 0:

        def xtrans(x):
            return x * (x_cut - x_min) + x_min

        def ytrans(y):
            return y * y_std + y_mean

        smp.plot_n_components()
        smp.plot_corners()
        smp.plot_observed(
            x_test,
            y_test,
            xtrans=xtrans,
            ytrans=ytrans,
            filename="fixed",
            y_true=y_true,
        )
        nlpd = smp.nlpd(x_test, y_test, ytrans, y_std)
        print(-1 * nlpd)
