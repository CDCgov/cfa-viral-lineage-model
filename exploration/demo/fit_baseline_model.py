#!/usr/bin/env python3

import sys

import jax
import numpy as np
import numpyro
import polars as pl
from numpyro.infer import MCMC, NUTS

import linmod.models as models

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
    .pivot(on="lineage", index=["fd_offset", "division"], values="count")
    .fill_null(0)
)

# Infer parameters

model = models.BaselineModel(
    divisions=data["division"],
    counts=data.select(sorted(data.columns)).drop("fd_offset", "division"),
)

mcmc = MCMC(
    NUTS(model.numpyro_model),
    num_samples=500,
    num_warmup=2500,
    num_chains=4,
)

mcmc.run(jax.random.key(0))

# Export samples

print(
    model.create_forecasts(mcmc, np.arange(-30, 15)).write_csv(),
    end="",
)
