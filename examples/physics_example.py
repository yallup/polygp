import os
import sys

import numpy as np
import pandas as pd
import tqdm
from mpi4py import MPI

from polygp import SpectralMixtureProcess

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import matplotlib.pyplot as plt

plt.style.use("computermodern.mplstyle")


def data_load(filename, datadir):
    df = pd.read_csv(os.path.join(datadir, filename))

    x = df["z"].to_numpy()
    y = df["H"].to_numpy()
    y_err = df["dH"].to_numpy()

    cut = int(len(x))

    x_cut = x[:cut].max()
    y_mean = y[:cut].mean()
    y_std = y[:cut].std()
    x_min = x.min()

    y = (y - y_mean) / y_std
    y_err = y_err / y_std
    # y=y-y_mean
    x = (x - x.min()) / (x_cut - x.min())

    x_train = x[:cut].squeeze()
    y_train = y[:cut].squeeze()
    x_test = x[cut:].squeeze()
    y_test = y[cut:].squeeze()

    return x_train, y_train, x_test, y_test, x_cut, y_mean, y_std, x_min, y_err


if __name__ == "__main__":
    base_dir = "chains"
    os.makedirs(base_dir, exist_ok=True)
    datadir = "data"
    nlive = 50

    (
        x_train,
        y_train,
        x_test,
        y_test,
        x_cut,
        y_mean,
        y_std,
        x_min,
        y_err,
    ) = data_load(f"cc.csv", "physics_data")
    x_test = np.asarray([0.0, 1.1, 0.0])
    y_test = np.asarray([0.0, 0.0, 0.0])
    smp = SpectralMixtureProcess(
        X=x_train,
        Y=y_train,
        kernel_n_max=4,
        base_dir=base_dir,
        file_root="cc",
        Y_err=y_err,
    )
    # smp.train(nlive=nlive,fac_repeat=5)
    # params = build_and_train_gp(x_train, y_train)
    if rank == 0:
        smp.plot_n_components()
        smp.plot_corners()

        def xtrans(x):
            return x * (x_cut - x_min) + x_min

        def ytrans(y):
            return y * y_std + y_mean

        smp.plot_observed(
            x_test, y_test, xtrans=xtrans, ytrans=ytrans, filename="cc", y_std=y_std
        )
        nlpd = smp.nlpd(x_test, y_test, ytrans, y_std)
        print(nlpd)
