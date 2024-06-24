#!/usr/bin/env python3

import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax.random import PRNGKey
from load_metadata import load_metadata
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)


def numpyro_model(
    counts: np.ndarray,
    divisions: np.ndarray,
    time: np.ndarray,
):
    """
    Observations are counts of lineages for each division-day.
    See https://doi.org/10.1101/2023.01.02.23284123

    counts:         A matrix of counts with shape (num_observations, num_lineages).
    divisions:      A vector of indices representing the division of each observation.
    time:           A vector of the time covariate for each observation.
    """

    num_lineages = counts.shape[1]
    num_divisions = np.unique(divisions).size

    # beta_0[g, l] is the intercept for lineage l in division g
    mu_beta_0 = numpyro.sample(
        "mu_beta_0",
        dist.StudentT(3, -5, 5),
        sample_shape=(num_lineages,),
    )
    sigma_beta_0 = numpyro.sample(
        "sigma_beta_0",
        dist.TruncatedNormal(2, 1, low=0),
        sample_shape=(num_lineages,),
    )
    z_0 = numpyro.sample(
        "z_0",
        dist.StudentT(2),
        sample_shape=(num_divisions, num_lineages),
    )
    beta_0 = numpyro.deterministic(
        "beta_0",
        mu_beta_0 + sigma_beta_0 * z_0,
    )

    # mu_beta_1[l] is the mean of the slope for lineage l
    mu_hierarchical = numpyro.sample(
        "mu_hierarchical",
        dist.Normal(-1, np.sqrt(0.5)),
    )
    sigma_hierarchical = numpyro.sample(
        "sigma_hierarchical",
        dist.TruncatedNormal(1, np.sqrt(0.1), low=0),
    )
    z_mu = numpyro.sample(
        "z_mu",
        dist.Normal(0, 1),
        sample_shape=(num_lineages,),
    )
    mu_beta_1 = numpyro.deterministic(
        "mu_beta_1",
        mu_hierarchical + sigma_hierarchical * z_mu,
    )

    # beta_1[g, l] is the slope for lineage l in division g
    sigma_beta_1 = numpyro.sample(
        "sigma_beta_1",
        dist.TruncatedNormal(0.5, 2, low=0),
        sample_shape=(num_lineages,),
    )
    Omega_decomposition = numpyro.sample(
        "Omega_decomposition",
        dist.LKJCholesky(num_lineages, 2),
    )
    # A faster version of `np.diag(sigma_beta_1) @ Omega_decomposition`
    Sigma_decomposition = sigma_beta_1[:, None] * Omega_decomposition
    z_1 = numpyro.sample(
        "z_1",
        dist.Normal(0, 1),
        sample_shape=(num_divisions, num_lineages),
    )
    beta_1 = numpyro.deterministic(
        "beta_1",
        mu_beta_1 + z_1 @ Sigma_decomposition.T,
    )

    # z[i, l] is the unnormalized probability of lineage l for observation i
    z = beta_0[divisions, :] + beta_1[divisions, :] * time[:, None]

    numpyro.sample(
        "Y",
        dist.Multinomial(total_count=counts.sum(axis=1), logits=z),
        obs=counts,
    )


if __name__ == "__main__":
    # Load the data

    URL = "https://data.nextstrain.org/files/ncov/open/north-america/metadata.tsv.xz"

    data = (
        load_metadata(url=URL, lineage_column_name="clade_nextstrain")
        .with_columns(pl.col("date").cast(pl.Date))
        .filter(pl.col("date") >= pl.col("date").max() - 90)
        .select(["lineage", "date", "count", "division"])
        .pivot(
            index=["date", "division"],
            columns="lineage",
            values="count",
        )
        .fill_null(0)
    )

    # Extract count matrix, division indices, and time covariate

    counts = data.drop(["date", "division"])
    counts = counts.select(sorted(counts.columns))

    divisions_key, divisions_encoded = np.unique(data["division"], return_inverse=True)

    time = (
        data.with_columns(
            (pl.col("date") - pl.col("date").min()).dt.total_days().alias("day"),
        )
        .with_columns((pl.col("day") - pl.col("day").mean()) / pl.col("day").std())[
            "day"
        ]
        .to_numpy()
    )

    # Infer parameters

    mcmc = MCMC(
        NUTS(numpyro_model),
        num_samples=500,
        num_warmup=2500,
        num_chains=4,
    )

    mcmc.run(
        PRNGKey(0),
        counts.to_numpy(),
        divisions_encoded,
        time,
    )

    # Export parameter samples

    beta_1 = np.asarray(mcmc.get_samples(group_by_chain=True)["beta_1"])
    n_chains, n_iterations, n_divisions, n_lineages = beta_1.shape

    regional_growth_rates = pl.DataFrame(
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
            "growth_rate": beta_1.flatten(),
        }
    )

    mu_beta_1 = np.asarray(mcmc.get_samples(group_by_chain=True)["mu_beta_1"])
    n_chains, n_iterations, n_lineages = mu_beta_1.shape

    global_growth_rates = pl.DataFrame(
        {
            "chain": np.repeat(np.arange(n_chains) + 1, n_iterations * n_lineages),
            "iteration": np.tile(
                np.repeat(np.arange(n_iterations) + 1, n_lineages), n_chains
            ),
            "division": "[Global]",
            "lineage": np.tile(counts.columns, n_chains * n_iterations),
            "growth_rate": mu_beta_1.flatten(),
        }
    )

    df = pl.concat([regional_growth_rates, global_growth_rates]).sort(
        ["chain", "iteration", "division", "lineage"]
    )

    print(df.write_csv())
