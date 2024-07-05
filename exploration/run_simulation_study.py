#!/usr/bin/env python3

# %%
from functools import reduce

import jax
import models
import numpy as np
import numpyro
import polars as pl
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.special import softmax
from utils import expand_grid, pl_softmax

# Load the real data. We will sample new counts

data = (
    pl.read_csv("metadata.csv")
    .cast({"date": pl.Date}, strict=False)
    .drop_nulls(subset=["date"])  # Drop dates that aren't resolved to the day
    .filter(pl.col("date") >= pl.col("date").max() - 90)
    # TODO: Remove for a more comprehensive sim study
    .filter(pl.col("lineage").str.starts_with("24"), division="California")
    .select("lineage", "date", "count", "division")
    .pivot(on="lineage", index=["date", "division"], values="count")
    .fill_null(0)
)

# Extract count matrix, division indices, and time covariate

counts = data.select(sorted(data.columns)).drop(["date", "division"])
division_names, divisions = np.unique(data["division"], return_inverse=True)
time = (data["date"] - data["date"].min()).dt.total_days().to_numpy()


def time_standardizer(t):
    return (t - time.mean()) / time.std()


# Fix true parameter values

prior_predictive = Predictive(models.independent_divisions_model, num_samples=1)

result = prior_predictive(
    jax.random.key(0),
    divisions,
    time_standardizer(time),
    N=counts.sum_horizontal().to_numpy(),
    num_lineages=counts.shape[1],
)
true_beta_0 = result["beta_0"][1]
true_beta_1 = result["beta_1"][1]

time7 = time_standardizer(time.max() + 7)

truth = expand_grid(
    division=division_names,
    lineage=counts.columns,
).with_columns(
    true_phi_time7=softmax(true_beta_0 + true_beta_1 * time7, axis=1).flatten(),
)

# Simulate new observed proportions and attempt to recover parameters

likelihood = numpyro.handlers.condition(
    Predictive(models.independent_divisions_model, num_samples=100),
    {
        # We must condition the unit-scaled parameters,
        # since we use numpyro's reparam
        "beta_0_decentered": (true_beta_0 + 5) / 2,
        "beta_1_decentered": (true_beta_1 + 1) / 1.8,
    },
)

sampled_counts_dfs = likelihood(
    jax.random.key(0),
    divisions,
    time_standardizer(time),
    N=counts.sum_horizontal().to_numpy(),
    num_lineages=counts.shape[1],
)

samples_dfs = []

for i, sampled_counts in enumerate(sampled_counts_dfs["Y"]):
    NUM_ITERATIONS = 500

    mcmc = MCMC(
        NUTS(models.independent_divisions_model),
        num_samples=NUM_ITERATIONS,
        num_warmup=500,
    )

    mcmc.run(
        jax.random.key(i),
        divisions,
        time_standardizer(time),
        counts=sampled_counts,
    )

    samples = expand_grid(
        sim_study_iteration=[i + 1],
        mcmc_iteration=np.arange(NUM_ITERATIONS) + 1,
        division=division_names,
        lineage=counts.columns,
    ).with_columns(
        beta_0=np.asarray(mcmc.get_samples()["beta_0"]).flatten(),
        beta_1=np.asarray(mcmc.get_samples()["beta_1"]).flatten(),
    )

    samples_dfs.append(samples)

posterior_medians = (
    reduce(lambda x, y: x.vstack(y), samples_dfs)
    .with_columns(
        phi_time7=pl_softmax(
            pl.col("beta_0") + pl.col("beta_1") * time7,
            over=["sim_study_iteration", "mcmc_iteration", "division"],
        )
    )
    .group_by("sim_study_iteration", "division", "lineage")
    .agg(pl.median("phi_time7"))
    .join(truth, on=["division", "lineage"])
    .sort("sim_study_iteration", "division", "lineage")
)

posterior_medians.write_csv("medians.csv")

# %%
