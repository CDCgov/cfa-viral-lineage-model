import polars as pl

from linmod.utils import pl_mae


def mae(samples, data):
    """
    A simple MAE on phi for each lineage-division-day.

    `samples` should have the standard model output format.
    `data` should have columns `(lineage, division, day, count)`.

    Returns a DataFrame with columns `(lineage, division, day, mae)`.
    """

    return (
        (
            data.with_columns(
                phi=(
                    pl.col("count") / pl.sum("count").over("division", "day")
                ),
            )
            .drop("count")
            .join(
                samples,
                on=("lineage", "division", "day"),
                how="left",
                suffix="_sampled",
            )
        )
        .group_by("lineage", "division", "day")
        .agg(mae=pl_mae("phi", "phi_sampled"))
    )
