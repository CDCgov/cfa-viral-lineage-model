#!/usr/bin/env python3

import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax.nn import softmax
from jax.random import PRNGKey
from load_metadata import load_metadata
from numpyro.infer import MCMC, NUTS


def numpyro_model(counts, N):
    parameter_shape = (counts.shape[1],)
    z = numpyro.sample("z", dist.Normal(0, 1), sample_shape=parameter_shape)
    numpyro.deterministic("phi", softmax(z))

    # Note that a "logit" is any unnormalized probability to be sent to softmax,
    # not specfically logit(phi)
    numpyro.sample(
        "Y_tl",
        dist.Multinomial(total_count=N, logits=z),
        obs=counts,
    )


if __name__ == "__main__":
    # Load the data

    data = load_metadata(
        url="https://data.nextstrain.org/files/ncov/open/north-america/metadata.tsv.xz",
        lineage_column_name="clade_nextstrain",
    ).filter(pl.col("date").str.starts_with("2024-05"))

    # Extract a count matrix

    counts = (
        data.select(["lineage", "date", "count"])
        .pivot(
            index="date", columns="lineage", values="count", aggregate_function="sum"
        )
        .fill_null(0)
        .select(pl.col("*").exclude("date"))
    )

    lineages = counts.columns
    counts = counts.to_numpy()
    N = counts.sum(axis=1)

    # Infer parameters

    mcmc = MCMC(
        NUTS(numpyro_model),
        num_samples=1000,
        num_warmup=200,
    )

    mcmc.run(PRNGKey(0), counts, N)

    # Export parameter samples

    phi_samples = np.asarray(mcmc.get_samples()["phi"])

    df = pl.DataFrame(phi_samples)
    df.columns = lineages
    df = df.with_columns(pl.arange(1, len(phi_samples) + 1).alias("iteration")).melt(
        id_vars=["iteration"], variable_name="lineage", value_name="phi"
    )

    print(df)
