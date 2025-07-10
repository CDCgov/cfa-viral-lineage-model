import numpy as np
import polars as pl
import pytest
from numpy.random import default_rng

from linmod.data import CountsFrame
from linmod.models import ForecastFrame
from linmod.utils import expand_grid


def _generate_fake_samples_and_data(
    rng,
    num_days,
    num_divisions,
    num_lineages,
    num_samples,
    sample_variance,
):
    r"""
    Generate fake data $Y_{tgl}$ and "forecasts" $phi_{tgl}$.

    - $Y_{tgl} \sim Uniform{0, ..., 99}$
    - Denote true proportion as $\pi_{tgl} = \frac{Y_{tgl}}{Y_{tg \cdot}}$
    - $phi_{tgl}' \sim N(\pi_{tgl}, \sigma^2); \phi_{tgl} = \frac{\phi_{tgl}'}{\Sum_{t',g'} \phi_{t'g'l}'}$
    """

    rng = default_rng(rng)

    # Generate fake population counts
    data = expand_grid(
        fd_offset=range(num_days),
        division=range(num_divisions),
        lineage=range(num_lineages),
    )

    data = data.with_columns(
        count=rng.integers(0, 100, data.shape[0]),
        date=pl.col("fd_offset"),
    )

    # Generate fake forecasts of population proportions
    samples = expand_grid(
        fd_offset=range(num_days),
        division=range(num_divisions),
        lineage=range(num_lineages),
        sample_index=range(num_samples),
    ).join(
        data.with_columns(
            phi_mean=(
                pl.col("count") / pl.sum("count").over("fd_offset", "division")
            ),
        ).drop("count"),
        on=("fd_offset", "division", "lineage"),
        how="left",
        suffix="_sampled",
    )

    samples = (
        samples.with_columns(
            phi=rng.normal(samples["phi_mean"], np.sqrt(sample_variance))
        )
        .drop("phi_mean")
        .with_columns(
            phi=pl.col("phi")
            / pl.sum("phi").over("sample_index", "fd_offset", "division")
        )
    )

    return samples, data


def test_countsframe():
    REQUIRED_COLUMNS = {
        "date",
        "fd_offset",
        "division",
        "lineage",
        "count",
    }

    rng = default_rng()

    # Ensure proper data passes validation
    _, data = _generate_fake_samples_and_data(
        rng,
        num_days=4,
        num_divisions=3,
        num_lineages=2,
        num_samples=1000,
        sample_variance=1.4,
    )

    CountsFrame(data).validate()

    # Ensure null counts fail validation
    data2 = data.with_columns(
        count=pl.when(pl.col("fd_offset") == 0)
        .then(None)
        .otherwise(pl.col("count"))
    )

    with pytest.raises(AssertionError):
        CountsFrame(data2).validate()

    # Ensure missing columns fail validation
    for col in REQUIRED_COLUMNS:
        with pytest.raises(AssertionError):
            CountsFrame(data.drop(col)).validate()

    # Ensure floating point counts fail validation
    with pytest.raises(AssertionError):
        CountsFrame(data.with_columns(count=pl.col("count") * 1.5)).validate()


def test_forecastframe():
    REQUIRED_COLUMNS = {
        "sample_index",
        "fd_offset",
        "division",
        "lineage",
        "phi",
    }

    rng = default_rng(0)

    # Ensure proper data passes validation
    samples, _ = _generate_fake_samples_and_data(
        rng,
        num_days=4,
        num_divisions=3,
        num_lineages=2,
        num_samples=1000,
        sample_variance=1.4,
    )

    ForecastFrame(samples).validate()

    # Ensure missing columns fail validation
    for col in REQUIRED_COLUMNS:
        with pytest.raises(AssertionError):
            CountsFrame(samples.drop(col)).validate()

    # Ensure improper proportions fail validation
    with pytest.raises(AssertionError):
        ForecastFrame(samples.with_columns(phi=pl.col("phi") * 1.5)).validate()
