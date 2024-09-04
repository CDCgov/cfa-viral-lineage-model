import numpy as np
import polars as pl


def _merge_samples_and_data(
    samples: pl.LazyFrame,
    data: pl.LazyFrame,
    seed: int = 42,
) -> pl.LazyFrame:

    rng = np.random.default_rng(seed)

    return (
        data.join(
            samples.rename({"phi": "phi_sampled"}),
            on=("fd_offset", "division", "lineage"),
            how="left",
        )
        .group_by("date", "fd_offset", "division", "sample_index")
        .agg(pl.col("lineage"), pl.col("phi_sampled"), pl.col("count"))
        .with_columns(
            count_sampled=pl.struct(
                pl.col("phi_sampled"), N=pl.col("count").list.sum()
            ).map_elements(
                lambda struct: list(
                    rng.multinomial(struct["N"], struct["phi_sampled"])
                ),
                return_dtype=pl.List(pl.Int64),
            )
        )
        .explode("lineage", "phi_sampled", "count", "count_sampled")
        .drop("phi_sampled")
    )
