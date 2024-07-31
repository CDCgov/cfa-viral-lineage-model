import polars as pl

from linmod.utils import pl_list_cycle, pl_norm


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


def proportions_mean_norm(sample, data, L=1) -> float:
    """The expected norm of phi error, summed over all divisions and days."""

    return (
        proportions_mean_norm_per_division_day(sample, data, L=L)
        .collect()
        .get_column("mean_norm")
        .sum()
    )


def proportions_energy_score_per_division_day(samples, data, L=2):
    """
    Monte Carlo approximation to the energy score (multivariate generalization of CRPS)
    of phi for each division-day.

    `samples` should have the standard model output format.
    `data` should have the standard model input format.

    Returns a DataFrame with columns `(division, fd_offset, energy_score)`.
    """

    # First, we will gather the values of phi' we will use for (phi-phi')
    samples = (
        samples.group_by("fd_offset", "division", "lineage")
        .agg(pl.col("sample_index"), pl.col("phi"))
        .with_columns(
            sample_index=pl_list_cycle(pl.col("sample_index"), 1),
            replicate=pl_list_cycle(pl.col("phi"), 1),
        )
        .explode("sample_index", "phi", "replicate")
    )

    return (
        _merge_samples_and_data(samples, data)
        .group_by("sample_index", "division", "fd_offset")
        .agg(
            term1=pl_norm(pl.col("phi") - pl.col("phi_sampled"), L),
            term2=pl_norm((pl.col("phi_sampled") - pl.col("replicate")), L),
        )
        .group_by("division", "fd_offset")
        .agg(
            energy_score=pl.col("term1").mean() - 0.5 * pl.col("term2").mean()
        )
    )


def proportions_energy_score(sample, data, L=2) -> float:
    """The energy score of phi, summed over all divisions and days."""

    return (
        proportions_energy_score_per_division_day(sample, data, L=L)
        .collect()
        .get_column("energy_score")
        .sum()
    )
