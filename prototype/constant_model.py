#!/usr/bin/env python3

import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax.nn import softmax
from jax.random import PRNGKey
from load_metadata import load_metadata
from numpyro.infer import MCMC, NUTS


def numpyro_model(counts):

    num_lineages = counts.shape[1]
    z = numpyro.sample("z", dist.Normal(0, 1), sample_shape=(num_lineages,))
    numpyro.deterministic("phi", softmax(z))

    # Note that a "logit" is any unnormalized probability to be sent to softmax,
    # not specfically logit(phi)
    numpyro.sample(
        "Y_",
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
        .select(["lineage", "date", "count"])
        .pivot(
            index="date",
            columns="lineage",
            values="count",
            aggregate_function="sum",
        )
        .fill_null(0)
    )

    # Extract count matrix

    counts = data.drop("date")

    # Infer parameters

    mcmc = MCMC(
        NUTS(numpyro_model),
        num_samples=1000,
        num_warmup=200,
    )

    mcmc.run(PRNGKey(0), counts.to_numpy())

    # Export parameter samples

    phi = np.asarray(mcmc.get_samples()["phi"])
    n_iterations, n_lineages = phi.shape

    df = pl.DataFrame(
        {
            "iteration": np.repeat(np.arange(1, n_iterations + 1), n_lineages),
            "lineage": np.tile(counts.columns, n_iterations),
            "phi": phi.flatten(),
        }
    )

    print(df.write_csv())
