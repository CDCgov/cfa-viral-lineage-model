#!/usr/bin/env python3

import sys

import jax
import numpy as np
import numpyro
import polars as pl
from numpyro.infer import MCMC, NUTS

import linmod.models as models
from linmod.utils import expand_grid, pl_softmax

numpyro.set_host_device_count(4)


# Load the data

if len(sys.argv) != 2:
    print(
        "Usage: python3 fit_indep_div_model.py <data_path> > samples.csv",
        file=sys.stderr,
    )
    sys.exit(1)

data = (
    pl.read_csv(sys.argv[1], try_parse_dates=True)
    .pivot(on="lineage", index=["fd_offset", "division"], values="count")
    .fill_null(0)
)

# Extract count matrix, division indices, and time covariate

counts = data.select(sorted(data.columns)).drop("fd_offset", "division")
division_names, divisions = np.unique(data["division"], return_inverse=True)
time = data["fd_offset"].to_numpy()


def time_standardizer(t):
    return (t - time.mean()) / time.std()


# Infer parameters

NUM_CHAINS = 4
NUM_ITERATIONS = 500

mcmc = MCMC(
    NUTS(models.independent_divisions_model),
    num_samples=NUM_ITERATIONS,
    num_warmup=2500,
    num_chains=NUM_CHAINS,
)

mcmc.run(
    jax.random.key(0),
    divisions,
    time_standardizer(time),
    counts=counts.to_numpy(),
)

# Collect posterior regression parameter samples

samples = (
    expand_grid(
        chain=np.arange(NUM_CHAINS),
        iteration=np.arange(NUM_ITERATIONS),
        division=division_names,
        lineage=counts.columns,
    )
    .with_columns(
        beta_0=np.asarray(mcmc.get_samples()["beta_0"]).flatten(),
        beta_1=np.asarray(mcmc.get_samples()["beta_1"]).flatten(),
        sample_index=pl.col("iteration")
        + pl.col("chain") * NUM_ITERATIONS
        + 1,
    )
    .drop("chain", "iteration")
)

# Compute posterior samples for population-level lineage proportions over time

print(
    expand_grid(
        sample_index=samples["sample_index"].unique(),
        fd_offset=np.arange(-30, 15),
    )
    .join(samples, on="sample_index")
    .with_columns(
        phi=pl_softmax(
            pl.col("beta_0")
            + pl.col("beta_1") * time_standardizer(pl.col("fd_offset")),
        ).over("sample_index", "division", "fd_offset")
    )
    .drop("beta_0", "beta_1")
    .write_csv(),
    end="",
)
