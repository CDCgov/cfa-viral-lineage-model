#!/usr/bin/env python3

import sys
from pathlib import Path

import jax
import numpy as np
import numpyro
import polars as pl
from numpyro.infer import MCMC, NUTS

import linmod.models as models
from linmod.utils import expand_grid, pl_softmax

numpyro.set_host_device_count(4)


# Load the data

data = (
    # TODO: add proper handling of missing path argument
    pl.read_csv(Path(sys.argv[1]))
    .cast({"date": pl.Date}, strict=False)
    .drop_nulls(subset=["date"])  # Drop dates that aren't resolved to the day
    .filter(pl.col("date") >= pl.col("date").max() - 90)
    .select(["lineage", "date", "count", "division"])
    .pivot(on="lineage", index=["date", "division"], values="count")
    .fill_null(0)
)

# Extract count matrix, division indices, and time covariate

counts = data.select(sorted(data.columns)).drop(["date", "division"])
division_names, divisions = np.unique(data["division"], return_inverse=True)
time = (data["date"] - data["date"].min()).dt.total_days().to_numpy()


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

samples = expand_grid(
    chain=np.arange(NUM_CHAINS) + 1,
    iteration=np.arange(NUM_ITERATIONS) + 1,
    division=division_names,
    lineage=counts.columns,
).with_columns(
    beta_0=np.asarray(mcmc.get_samples()["beta_0"]).flatten(),
    beta_1=np.asarray(mcmc.get_samples()["beta_1"]).flatten(),
)

# Compute posterior samples for population-level lineage proportions

for delta_t in range(-30, 15):
    t = time_standardizer(time.max() + delta_t)

    samples = samples.with_columns(
        pl_softmax(
            pl.col("beta_0") + pl.col("beta_1") * t,
            over=["chain", "iteration", "division"],
        ).alias(f"phi_t{delta_t}".replace("-", "m")),
    )

samples = samples.drop(["beta_0", "beta_1"])

print(samples.write_csv())
