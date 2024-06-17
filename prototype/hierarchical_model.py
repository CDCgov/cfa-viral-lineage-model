#!/usr/bin/env python3

import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax.random import PRNGKey
from load_metadata import load_metadata
from numpyro.infer import MCMC, NUTS


def numpyro_model(
    counts: np.ndarray,
    regions: np.ndarray,
    days: np.ndarray,
):
    """
    Observations are counts of lineages for each region-day.
    See https://doi.org/10.1101/2023.01.02.23284123

    counts:         A matrix of counts with shape (num_observations, num_lineages).
    regions:        A vector of indices representing the region of each observation.
    days:           A vector of the time covariate for each observation.
    """

    num_lineages = counts.shape[1]
    num_regions = len(np.unique(regions))

    # beta_0[g, l] is the intercept for lineage l in region g
    beta_0 = numpyro.sample(
        "beta_0",
        dist.Normal(0, 1),
        sample_shape=(num_regions, num_lineages),
    )

    # mu_beta1[l] is the mean of the slope for lineage l
    mu_beta1 = numpyro.sample(
        "mu_beta1",
        dist.Normal(0, 1),
        sample_shape=(num_lineages,),
    )

    # beta_1[g, l] is the slope for lineage l in region g
    beta_1 = numpyro.sample(
        "beta_1",
        dist.Normal(mu_beta1, 1),
        sample_shape=(num_regions,),
    )

    # z[g, l] is the unnormalized probability of lineage l for observation g
    z = beta_0[regions, :] + beta_1[regions, :] * days[:, None]

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
        .filter(pl.col("date").str.starts_with("2024-05"))
        .select(["lineage", "date", "count", "division"])
        .pivot(
            index=["date", "division"],
            columns="lineage",
            values="count",
        )
        .fill_null(0)
    )

    # Extract count matrix, regions, and time covariate

    counts = data.drop(["date", "division", "day"])

    regions_key, regions_encoded = np.unique(data["division"], return_inverse=True)

    days = data.with_columns(
        (pl.col("date").cast(pl.Date) - pl.date(2024, 5, 1))
        .dt.total_days()
        .alias("day"),
    )["day"].to_numpy()

    # Infer parameters

    mcmc = MCMC(
        NUTS(numpyro_model),
        num_samples=1000,
        num_warmup=200,
    )

    mcmc.run(
        PRNGKey(0),
        counts.to_numpy(),
        regions_encoded,
        days,
    )

    # Export parameter samples

    beta1 = np.asarray(mcmc.get_samples()["beta_1"])
    n_iterations, n_regions, n_lineages = beta1.shape

    regional_growth_rates = pl.DataFrame(
        {
            "iteration": np.repeat(
                np.arange(1, n_iterations + 1), n_regions * n_lineages
            ),
            "region": np.tile(np.repeat(regions_key, n_lineages), n_iterations),
            "lineage": np.tile(np.tile(counts.columns, n_regions), n_iterations),
            "growth_rate": beta1.flatten(),
        }
    )

    mu_beta1 = np.asarray(mcmc.get_samples()["mu_beta1"])
    n_iterations, n_lineages = mu_beta1.shape

    global_growth_rates = pl.DataFrame(
        {
            "iteration": np.repeat(np.arange(1, n_iterations + 1), n_lineages),
            "region": "[Global]",
            "lineage": np.tile(counts.columns, n_iterations),
            "growth_rate": mu_beta1.flatten(),
        }
    )

    df = (
        pl.concat([regional_growth_rates, global_growth_rates])
        .sort(["iteration", "region", "lineage"])
        .with_columns(fitness_advantage=((pl.col("growth_rate") * 7).exp() - 1))
    )

    print(df.write_csv())
