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
        "Usage: python3 fit_baseline_model.py <data_path> > samples.csv",
        file=sys.stderr,
    )
    sys.exit(1)

data = (
    pl.read_csv(sys.argv[1], try_parse_dates=True)
    .pivot(on="lineage", index=["lcd_offset", "division"], values="count")
    .fill_null(0)
)

# Extract count matrix, division indices, and time covariate

counts = data.select(sorted(data.columns)).drop("lcd_offset", "division")
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
        lcd_offset=np.arange(-30, 15),
    )
    .join(samples, on="sample_index")
    .with_columns(
        phi=pl_softmax(pl.col("logit_phi")).over(
            "sample_index", "division", "lcd_offset"
        )
    )
    .drop("logit_phi")
    .write_csv(),
    end="",
)
