import polars as pl

from linmod.utils import pl_mae


def proportions_mae_per_division_day(samples, data) -> pl.DataFrame:
    """
    A simple MAE on phi for each lineage-division-day.

    `samples` should have the standard model output format.
    `data` should have the standard model input format.

    Returns a DataFrame with columns `(lineage, division, fd_offset, mae)`.
    """

    return (
        (
            data.with_columns(
                phi=(
                    pl.col("count")
                    / pl.sum("count").over("division", "fd_offset")
                ),
            )
            .drop("count")
            .join(
                samples,
                on=("lineage", "division", "fd_offset"),
                how="left",
                suffix="_sampled",
            )
        )
        .group_by("lineage", "division", "fd_offset")
        .agg(mae=pl_mae("phi", "phi_sampled"))
        .group_by("division", "fd_offset")
        .agg(pl.sum("mae"))
    )


def proportions_mae(
    sample: pl.DataFrame, data: pl.DataFrame, score_column: str = "mae"
) -> float:
    """MAE on phi, summed over all lineages, divisions, and days"""

    return (
        proportions_mae_per_division_day(sample, data)
        .collect()
        .get_column(score_column)
        .sum()
    )
