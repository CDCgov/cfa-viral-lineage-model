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

if len(sys.argv) != 2:
    print(
        "Usage: python3 fit_baseline_model.py <data_path> > samples.csv",
        file=sys.stderr,
    )
    sys.exit(1)

data = (
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


# Infer parameters

NUM_CHAINS = 4
NUM_ITERATIONS = 500

mcmc = MCMC(
    NUTS(models.baseline_model),
    num_samples=NUM_ITERATIONS,
    num_warmup=2500,
    num_chains=NUM_CHAINS,
)

mcmc.run(
    jax.random.key(0),
    divisions,
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
        logit_phi=np.asarray(mcmc.get_samples()["logit_phi"]).flatten(),
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
        day=np.arange(-30, 15),
    )
    .join(samples, on="sample_index")
    .with_columns(
        phi=pl_softmax(pl.col("logit_phi")).over(
            "sample_index", "division", "day"
        )
    )
    .drop("logit_phi")
    .write_csv()
)
