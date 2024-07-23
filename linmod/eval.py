import polars as pl

from linmod.utils import pl_norm


def _merge_samples_and_data(samples, data):
    return (
        data.with_columns(
            phi=(
                pl.col("count") / pl.sum("count").over("division", "fd_offset")
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


def proportions_mean_norm_per_division_day(samples, data, L=1):
    """
    The expected norm of phi error for each division-day.

    `samples` should have the standard model output format.
    `data` should have the standard model input format.

    Returns a DataFrame with columns `(division, fd_offset, mean_norm)`.
    """

    return (
        _merge_samples_and_data(samples, data)
        .group_by("sample_index", "division", "fd_offset")
        .agg(norm=pl_norm(pl.col("phi") - pl.col("phi_sampled"), L))
        .group_by("division", "fd_offset")
        .agg(mean_norm=pl.mean("norm"))
    )


def proportions_energy_score_per_division_day(samples, data):
    """
    Monte Carlo approximation to the energy score (multivariate generalization of CRPS)
    of phi for each division-day.

    `samples` should have the standard model output format.
    `data` should have the standard model input format.

    Returns a DataFrame with columns `(division, fd_offset, energy_score)`.
    """

    return (
        _merge_samples_and_data(samples, data)
        .with_columns(
            # Gather the values of X' we will use for (X-X')
            replicate=pl.col("phi_sampled")
            .shift(1)
            .over("fd_offset", "division", "lineage"),
        )
        .group_by("sample_index", "division", "fd_offset")
        .agg(
            term1=pl_norm(pl.col("phi") - pl.col("phi_sampled"), 2),
            # Note that the expected value for term2 is over n-1 pairs,
            # and the nth pair has replicate==null. However, polars will
            # silently drop the null, resulting in term2==0 for the nth pair.
            # To avoid this, we force term2 to null for the nth pair, so
            # that the expected value drops the null and is taken over n-1 samples.
            term2=pl.when(pl.col("replicate").has_nulls())
            .then(None)
            .otherwise(
                pl_norm((pl.col("phi_sampled") - pl.col("replicate")), 2)
            ),
        )
        .group_by("division", "fd_offset")
        .agg(
            energy_score=pl.col("term1").mean() - 0.5 * pl.col("term2").mean()
        )
    )
