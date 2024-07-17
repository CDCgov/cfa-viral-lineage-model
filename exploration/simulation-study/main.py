#!/usr/bin/env python3

# %%
import sys

import jax
import numpy as np
import polars as pl
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.special import softmax

import linmod.models as models
from linmod.utils import expand_grid, pl_softmax

# Load the real data. We will sample new counts

if len(sys.argv) != 2:
    print("Usage: python3 main.py <data_path>", file=sys.stderr)
    sys.exit(1)

data = (
    pl.read_csv(sys.argv[1])
    .filter(
        pl.col("division").is_in(
            ["Arizona", "California", "New York", "Pennsylvania"]
        ),
    )
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

prior_predictive = Predictive(
    models.independent_divisions_model, num_samples=1
)

result = prior_predictive(
    jax.random.key(0),
    divisions,
    time_standardizer(time),
    N=counts.sum_horizontal().to_numpy() * 100,
    num_lineages=counts.shape[1],
)
true_beta_0 = result["beta_0"][1]
true_beta_1 = result["beta_1"][1]

present_time = time_standardizer(time.max())

truth = expand_grid(
    division=division_names,
    lineage=counts.columns,
).with_columns(
    true_phi_present=softmax(
        true_beta_0 + true_beta_1 * present_time, axis=1
    ).flatten(),
)

# Simulate new observed proportions and attempt to recover parameters

likelihood = models.multinomial_likelihood(
    true_beta_0,
    true_beta_1,
    divisions,
    time_standardizer(time),
    counts.sum_horizontal().to_numpy() * 500,
)

# %%

NUM_MCMC_ITERATIONS = 500
NUM_SIM_STUDY_ITERATIONS = 50

keys = jax.random.split(jax.random.key(0), NUM_SIM_STUDY_ITERATIONS)
samples_dfs = []

for i in range(NUM_SIM_STUDY_ITERATIONS):
    sample_key, run_key = jax.random.split(keys[i], 2)

    sampled_counts = likelihood.sample(sample_key)

    mcmc = MCMC(
        NUTS(models.independent_divisions_model),
        num_samples=NUM_MCMC_ITERATIONS,
        num_warmup=500,
    )

    mcmc.run(
        run_key, divisions, time_standardizer(time), counts=sampled_counts
    )

    samples = expand_grid(
        sim_study_iteration=[i + 1],
        mcmc_iteration=np.arange(NUM_MCMC_ITERATIONS) + 1,
        division=division_names,
        lineage=counts.columns,
    ).with_columns(
        beta_0=np.asarray(mcmc.get_samples()["beta_0"]).flatten(),
        beta_1=np.asarray(mcmc.get_samples()["beta_1"]).flatten(),
    )

    samples_dfs.append(samples)

# Export posterior samples

print(
    pl.concat(samples_dfs)
    .with_columns(
        phi_present=pl_softmax(
            pl.col("beta_0") + pl.col("beta_1") * present_time
        ).over("sim_study_iteration", "mcmc_iteration", "division")
    )
    .join(truth, on=["division", "lineage"])
    .sort("sim_study_iteration", "mcmc_iteration", "division", "lineage")
    .drop("beta_0", "beta_1")
    .write_csv()
)

# %%
