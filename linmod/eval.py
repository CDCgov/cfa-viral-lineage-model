import polars as pl

from linmod.utils import pl_mae


def proportions_mae_per_division_day(samples, data) -> pl.DataFrame:
    """
    A simple MAE on phi for each lineage-division-day.

    `samples` should have the standard model output format.
    `data` should have columns `(lineage, division, day, count)`.

    Returns a DataFrame with columns `(lineage, division, day, mae)`.
    """

    return (
        (
            data.with_columns(
                phi=(pl.col("count") / pl.sum("count").over("division", "day")),
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


def proportions_mae(
    sample: pl.DataFrame, data: pl.DataFrame, score_column: str = "mae"
) -> float:
    """MAE on phi, summed over all lineages, divisions, and days"""

    return proportions_mae(sample, data).collect().get_columns(score_column).sum()
