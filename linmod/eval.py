import polars as pl

from linmod.utils import pl_list_cycle, pl_norm


def _merge_samples_and_data(samples, data):
    r"""
    Join the forecast samples and raw data dataframes, assuming they have the
    standard format.

    Also compute the true proportions from the raw data.
    """
    result = (
        data.with_columns(
            phi=(
                pl.col("count") / pl.sum("count").over("fd_offset", "division")
            ),
        )
        .drop("count")
        .join(
            samples,
            on=("fd_offset", "division", "lineage"),
            how="left",
            suffix="_sampled",
        )
    )

    return result


def proportions_mean_norm_per_division_day(samples, data, p=1):
    r"""
    The expected norm of proportion forecast error for each division-day.

    $E[ || f_{tg} - \phi_{tg} ||_p ]$

    `samples` should have the standard model output format.
    `data` should have the standard model input format.

    Returns a DataFrame with columns `(division, fd_offset, mean_norm)`.
    """

    return (
        _merge_samples_and_data(samples, data)
        .group_by("fd_offset", "division", "sample_index")
        .agg(norm=pl_norm(pl.col("phi") - pl.col("phi_sampled"), p))
        .group_by("fd_offset", "division")
        .agg(mean_norm=pl.mean("norm"))
    )


def proportions_mean_norm(sample, data, p=1) -> float:
    r"""
    The expected norm of proportion forecast error, summed over all divisions and days.

    $\sum_{t, g} E[ || f_{tg} - \phi_{tg} ||_p ]$
    """

    return (
        proportions_mean_norm_per_division_day(sample, data, p=p)
        .collect()
        .get_column("mean_norm")
        .sum()
    )


def proportions_energy_score_per_division_day(samples, data, p=2):
    r"""
    Monte Carlo approximation to the energy score (multivariate generalization of CRPS)
    of proportion forecasts for each division-day.

    $E[ || f_{tg} - \phi_{tg} ||_p ] - \frac{1}{2} E[ || f_{tg} - \f_{tg}' ||_p ]$

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
        .group_by("fd_offset", "division", "sample_index")
        .agg(
            term1=pl_norm(pl.col("phi") - pl.col("phi_sampled"), p),
            term2=pl_norm((pl.col("phi_sampled") - pl.col("replicate")), p),
        )
        .group_by("fd_offset", "division")
        .agg(
            energy_score=pl.col("term1").mean() - 0.5 * pl.col("term2").mean()
        )
    )


def proportions_energy_score(sample, data, p=2) -> float:
    r"""
    The energy score of proportion forecasts, summed over all divisions and days.

    $$
    \sum_{t, g} E[ || f_{tg} - \phi_{tg} ||_p ]
    - \frac{1}{2} E[ || f_{tg} - f_{tg}' ||_p ]
    $$
    """

    return (
        proportions_energy_score_per_division_day(sample, data, p=p)
        .collect()
        .get_column("energy_score")
        .sum()
    )
