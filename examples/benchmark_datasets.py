import os
import sys

import jax
import numpy as np
import pandas as pd
import tqdm
from mpi4py import MPI

from polygp import (
    NonStationarySpectralMixtureProcess,
    SparseSpectralMixtureProcess,
    SpectralMixtureProcess,
    StaticSpectralMixtureProcess,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from matplotlib import pyplot as plt

plt.style.use("computermodern.mplstyle")


def data_load(filename, datadir):
    df = pd.read_csv(os.path.join(datadir, filename))

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    cut = int(len(x) * 0.6)

    x_cut = x[:cut].max()
    y_mean = y[:cut].mean()
    y_std = y[:cut].std()
    x_min = x.min()

    y = (y - y_mean) / y_std
    # y=y-y_mean
    x = (x - x.min()) / (x_cut - x.min())

    x_train = x[:cut].squeeze()
    y_train = y[:cut].squeeze()
    x_test = x[cut:].squeeze()
    y_test = y[cut:].squeeze()

    return x_train, y_train, x_test, y_test, x_cut, y_mean, y_std, x_min


if __name__ == "__main__":
    base_dir = "chains_ours"
    os.makedirs(base_dir, exist_ok=True)
    datadir = "data"
    nlive = 200
    results_file = open(os.path.join(base_dir, "results.txt"), "w")
    filenames = sorted(
        os.listdir(datadir),
        key=lambda x: int(x.split("-")[0])
        if x.split("-")[0].isdigit()
        else sys.maxsize,
    )
    for filename in tqdm.tqdm(filenames):
        if os.path.isfile(os.path.join(datadir, filename)):
            if "radio" not in filename:
                continue
            # print(filename)
            if "csv" in filename:
                cleaned_filename = filename.split("-")[1].split(".")[0]
                print(cleaned_filename)
                (
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    x_cut,
                    y_mean,
                    y_std,
                    x_min,
                ) = data_load(filename, datadir)
                smp = SpectralMixtureProcess(
                    # smp =StaticSpectralMixtureProcess(
                    X=x_train,
                    Y=y_train,
                    kernel_n_max=6,
                    base_dir=base_dir,
                    file_root=cleaned_filename,
                )
                # with jax.disable_jit():
                output = smp.train(nlive=nlive, fac_repeat=2)
                # params = build_and_train_gp(x_train, y_train)
                if rank == 0:
                    smp.plot_corners()
                    smp.plot_n_components()

                    def xtrans(x):
                        return x * (x_cut - x_min) + x_min

                    def ytrans(y):
                        return y * y_std + y_mean

                    smp.plot_observed(
                        # np.asarray([1.1, 10.0]),
                        x_test,
                        y_test,
                        # np.asarray([0.0, 0.0]),
                        xtrans=xtrans,
                        ytrans=ytrans,
                        filename=cleaned_filename,
                    )
                    nlpd = smp.nlpd(x_test, y_test, ytrans, y_std)
                    print(nlpd)
                    results_file.write(
                        f"{cleaned_filename}: {nlpd:.2f} \t {output.nlike:.2e} \t {output.logZ :.2f}\n"
                    )
    results_file.close()
