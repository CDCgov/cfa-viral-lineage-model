#!/usr/bin/env python3

import jax
import models
import numpy as np
import numpyro
import polars as pl
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)


# Load the data

data = (
    pl.read_csv("metadata.csv")
    .with_columns(pl.col("date").cast(pl.Date, strict=False))
    .drop_nulls(subset=["date"])  # Drop dates that aren't resolved to the day
    .filter(pl.col("date") >= pl.col("date").max() - 90)
    .select(["lineage", "date", "count", "division"])
    .pivot(
        index=["date", "division"],
        on="lineage",
        values="count",
    )
    .fill_null(0)
)

# Extract count matrix, division indices, and time covariate

counts = data.drop(["date", "division"])
counts = counts.select(sorted(counts.columns))

divisions_key, divisions_encoded = np.unique(data["division"], return_inverse=True)

time = data.with_columns(
    (pl.col("date") - pl.col("date").min()).dt.total_days().alias("day"),
)["day"]
time_mean, time_std = time.mean(), time.std()
time_standardized = ((time - time_mean) / time_std).to_numpy()

# Infer parameters

mcmc = MCMC(
    NUTS(models.independent_divisions_model),
    num_samples=500,
    num_warmup=2500,
    num_chains=4,
)

mcmc.run(
    jax.random.key(0),
    counts.to_numpy(),
    divisions_encoded,
    time_standardized,
    extra_fields=("potential_energy",),
)

# Collect posterior regression parameter samples

beta_0 = np.asarray(mcmc.get_samples(group_by_chain=True)["beta_0"])
beta_1 = np.asarray(mcmc.get_samples(group_by_chain=True)["beta_1"])
potential_energy = np.asarray(
    mcmc.get_extra_fields(group_by_chain=True)["potential_energy"]
)
n_chains, n_iterations, n_divisions, n_lineages = beta_1.shape

df = pl.DataFrame(
    {
        "chain": np.repeat(
            np.arange(n_chains) + 1, n_iterations * n_divisions * n_lineages
        ),
        "iteration": np.tile(
            np.repeat(np.arange(n_iterations) + 1, n_divisions * n_lineages),
            n_chains,
        ),
        "division": np.tile(
            np.repeat(divisions_key, n_lineages), n_chains * n_iterations
        ),
        "lineage": np.tile(counts.columns, n_chains * n_iterations * n_divisions),
        "beta_0": beta_0.flatten(),
        "beta_1": beta_1.flatten(),
    }
).join(
    pl.DataFrame(
        {
            "chain": np.repeat(np.arange(n_chains) + 1, n_iterations),
            "iteration": np.tile(np.arange(n_iterations) + 1, n_chains),
            "potential_energy": potential_energy.flatten(),
        }
    ),
    on=["chain", "iteration"],
)

# Compute posterior samples for population-level lineage proportions

for delta_t in range(-30, 15):
    t_standardized = (time.max() + delta_t - time_mean) / time_std

    df = df.with_columns(
        (pl.col("beta_0") + pl.col("beta_1") * t_standardized).exp().alias("exp_z")
    ).with_columns(
        (
            pl.col("exp_z")
            / pl.col("exp_z").sum().over(["chain", "iteration", "division"])
        ).alias(f"phi_t{delta_t}".replace("-", "m")),
    )

df = df.drop(["beta_0", "beta_1", "exp_z"])

print(df.write_csv())
