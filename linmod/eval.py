from typing import Callable

import numpy as np
import polars as pl

from linmod.models import predict_counts
from linmod.utils import expand_grid, expand_phi, pl_list_cycle, pl_norm


def _merge_samples_and_data(data, samples, samples_are_phi: bool):
    r"""
    Join the forecast samples and raw data dataframes, assuming they have the
    standard format.

    Also compute the true proportions from the raw data.
    """
    if samples_are_phi:
        result = (
            data.with_columns(
                observed=(
                    pl.col("count")
                    / pl.sum("count").over("fd_offset", "division")
                ),
            )
            .drop("count")
            .join(
                samples,
                on=("fd_offset", "division", "lineage"),
                how="left",
            )
            .rename({"phi": "sampled"})
        )
    else:
        result = data.rename({"count": "observed"}).join(
            samples.rename({"count": "sampled"}),
            on=("fd_offset", "division", "lineage"),
            how="left",
        )

    return result


def generate_eval_counts(
    data,
    samples,
    fd_min,
    fd_max,
    num_samples: int | None = None,
    seed: int = 42,
):
    """
    Using the posterior distribution of population proportions in `samples` and the observed per-division-day counts in `data`, generate the forecast-predictive distribution on counts.

    samples (pl.DataFrame):        Posterior samples of population proportions in the standard
                                   model output format
    data (pl.DataFrame):           Count data in the standard model input format. This should be
                                   the data against which the predictive counts are to be scored.
    num_samples (optional int):    If specified, the posterior samples will be thinned to
                                   produce this many samples of counts. Ignored if larger
                                   than the number of posterior samples.
    seed (int):                    Seed for random number generation.
    """
    if isinstance(samples, pl.LazyFrame):
        samples = samples.collect()
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    lineages = samples["lineage"].unique().sort().to_list()

    # Drop missing division-days
    all_times = np.array(range(fd_min, fd_max + 1))
    division_names = samples["division"].unique().sort().to_list()
    num_gt = len(all_times) * len(division_names)

    data = data.filter(pl.col("division").is_in(division_names))
    obs_gt = (
        expand_grid(
            division=division_names,
            fd_offset=all_times,
        )
        .with_columns(index=pl.int_range(num_gt))
        .join(
            (
                data.group_by(["fd_offset", "division"])
                .agg(pl.col("count").sum())
                .sort("division", "fd_offset")
                .select("fd_offset", "division", "count")
            ),
            how="left",
            on=["fd_offset", "division"],
            validate="1:1",
        )
        .filter(pl.col("count").is_not_null())
    )

    slicer = tuple(obs_gt["index"])
    n = np.array(obs_gt["count"].to_list())
    division_days = obs_gt.select("division", "fd_offset")

    phi = expand_phi(samples)[:, slicer, :]

    niter = len(samples["sample_index"].unique())
    if num_samples is not None and num_samples < niter:
        keep = np.round(np.linspace(0, niter - 1, num_samples), 0).astype(int)
        phi = phi[keep, :, :]

    counts = (
        pl.concat(
            [
                pl.from_numpy(
                    np.array(
                        predict_counts(phi_i, n, seed + i)
                    ),  # Can't create from JAX array
                    schema=lineages,
                ).with_columns(division_days, sample_index=pl.lit(keep[i]))
                for phi_i, i in zip(phi, range(phi.shape[0]))
            ]
        )
        .unpivot(
            index=["sample_index", "division", "fd_offset"],
            variable_name="lineage",
            value_name="count",
        )
        .cast({"count": pl.Int64})
    )

    assert counts["count"].sum() == phi.shape[0] * n.sum()
    return counts


def score(
    fun: Callable,
    data: pl.LazyFrame,
    samples: pl.LazyFrame,
    samples_are_phi: bool,
    agg="sum_all",
    **kwargs
) -> float:
    if agg != "sum_all":
        raise RuntimeError(
            "I don't know how to aggregate scores except for agg='sum_all'."
        )
    return (
        fun(data, samples, samples_are_phi, **kwargs)
        .collect()
        .get_column("score")
        .sum()
    )


def mean_norm_per_division_day(data, samples, samples_are_phi, **kwargs):
    r"""
    The expected norm of forecast error for each division-day.

    $E[ || f_{tg} - \phi_{tg} ||_p ]$

    `samples` should have the standard model output format.
    `data` should have the standard model input format.

    Returns a DataFrame with columns `(division, fd_offset, mean_norm)`.
    """
    p = kwargs["p"] if "p" in kwargs.keys() else 1
    return (
        _merge_samples_and_data(data, samples, samples_are_phi)
        .group_by("fd_offset", "division", "sample_index")
        .agg(norm=pl_norm(pl.col("observed") - pl.col("sampled"), p))
        .group_by("fd_offset", "division")
        .agg(score=pl.mean("norm"))
    )


def energy_score_per_division_day(data, samples, samples_are_phi, **kwargs):
    r"""
    Monte Carlo approximation to the energy score (multivariate generalization of CRPS)
    of forecasts for each division-day.

    $E[ || f_{tg} - \phi_{tg} ||_p ] - \frac{1}{2} E[ || f_{tg} - \f_{tg}' ||_p ]$

    `samples` should have the standard model output format.
    `data` should have the standard model input format.

    Returns a DataFrame with columns `(division, fd_offset, energy_score)`.
    """
    p = kwargs["p"] if "p" in kwargs.keys() else 2
    if samples_are_phi:
        col = "phi"
    else:
        col = "count"

    # First, we will gather the values of phi' we will use for (phi-phi')
    samples = (
        samples.group_by("fd_offset", "division", "lineage")
        .agg(pl.col("sample_index"), pl.col(col))
        .with_columns(
            sample_index=pl_list_cycle(pl.col("sample_index"), 1),
            replicate=pl_list_cycle(pl.col(col), 1),
        )
        .explode("sample_index", col, "replicate")
    )

    return (
        _merge_samples_and_data(data, samples, samples_are_phi)
        .group_by("fd_offset", "division", "sample_index")
        .agg(
            term1=pl_norm(pl.col("observed") - pl.col("sampled"), p),
            term2=pl_norm((pl.col("sampled") - pl.col("replicate")), p),
        )
        .group_by("fd_offset", "division")
        .agg(score=pl.col("term1").mean() - 0.5 * pl.col("term2").mean())
    )
