from datetime import date

import numpy as np
import polars as pl
from numpy.random import default_rng
from polars.testing import assert_frame_equal

from linmod import eval
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
    - $phi_{tgl} \sim N(\pi_{tgl}, \sigma^2)$
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

    samples = samples.with_columns(
        phi=rng.normal(samples["phi_mean"], np.sqrt(sample_variance))
    ).drop("phi_mean")

    return samples, data


def test_proportions_mean_L1_norm(
    num_samples=100000, sample_variance=1.4, atol=0.05
):
    r"""
    Test the estimate of the expected L1 norm of phi error.

    The setup is as follows:
    - Generate fake data $Y_{tgl} \sim Uniform{0, ..., 99}$
    - Denote true proportion as $\phi_{tgl} = \frac{Y_{tgl}}{Y_{tg \cdot}}$
    - Generate fake "forecasts" $f_{tgl} \sim N(\phi_{tgl}, \sigma^2)$
    - Check $\sum_{t, g} E[ || f_{tg} - \phi_{tg} ||_1 ]$
      as reported by `eval.proportions_energy_score`

    Note that our "forecasts" are not actually proportions, but independent normal
    random variables. This is so that the quantity can be computed analytically.
    """

    rng = default_rng()

    NUM_DAYS = 4
    NUM_DIVISIONS = 3
    NUM_LINEAGES = 2

    samples, data = _generate_fake_samples_and_data(
        rng,
        num_days=NUM_DAYS,
        num_divisions=NUM_DIVISIONS,
        num_lineages=NUM_LINEAGES,
        num_samples=num_samples,
        sample_variance=sample_variance,
    )

    # Because we use L1 norm, this metric is equal to the sum of each component's MAE
    # from its mean.
    # The mean absolute error of a normal random variable from its mean is
    # $\sigma \sqrt{\frac{2}{\pi}}$ (because $X - \mu \sim N(0, \sigma^2)$, so
    # $|X - \mu| \sim \text{half-normal}(\sigma)$, which has this mean).

    assert np.isclose(
        eval.ProportionsEvaluator(samples, data).mean_norm(p=1),
        np.sqrt(sample_variance * 2 / np.pi)
        * NUM_DAYS
        * NUM_DIVISIONS
        * NUM_LINEAGES,
        atol=atol,
    )


def test_proportions_mean_L1_norm2():
    samples, data = _generate_fake_samples_and_data(
        None,
        num_days=2,
        num_divisions=2,
        num_lineages=2,
        num_samples=2,
        sample_variance=0,
    )

    # Put in fixed values for counts and phi
    data = data.sort("fd_offset", "division", "lineage").with_columns(
        count=pl.Series([1, 2, 3, 4, 5, 6, 7, 8]),
    )

    samples = samples.sort(
        "fd_offset", "division", "sample_index", "lineage"
    ).with_columns(
        phi=pl.Series(
            [
                0.2,
                0.8,
                0.3,
                0.7,
                0.4,
                0.6,
                0.5,
                0.5,
                0.2,
                0.8,
                0.3,
                0.7,
                0.4,
                0.6,
                0.5,
                0.5,
            ]
        ),
    )

    correct_output = pl.DataFrame(
        {
            "fd_offset": [0, 0, 1, 1],
            "division": [0, 1, 0, 1],
            "mean_norm": [
                np.abs(np.array([0.2, 0.3]) - 1 / (1 + 2)).mean()
                + np.abs(np.array([0.8, 0.7]) - 2 / (1 + 2)).mean(),
                np.abs(np.array([0.4, 0.5]) - 3 / (3 + 4)).mean()
                + np.abs(np.array([0.6, 0.5]) - 4 / (3 + 4)).mean(),
                np.abs(np.array([0.2, 0.3]) - 5 / (5 + 6)).mean()
                + np.abs(np.array([0.8, 0.7]) - 6 / (5 + 6)).mean(),
                np.abs(np.array([0.4, 0.5]) - 7 / (7 + 8)).mean()
                + np.abs(np.array([0.6, 0.5]) - 8 / (7 + 8)).mean(),
            ],
        }
    )

    result = (
        eval.ProportionsEvaluator(samples, data)
        ._mean_norm_per_division_day(p=1)
        .collect()
    )

    assert_frame_equal(
        result,
        correct_output,
        check_row_order=False,
        check_column_order=True,
        rtol=0,
        atol=0,
    )


def test_uncoverage():
    df = pl.concat(
        [
            pl.DataFrame(
                {
                    "date": date(1, 1, 1),
                    "fd_offset": [0] * 10,
                    "division": [0] * 10,
                    "lineage": ["A"] * 10,
                    "count": [10] * 10,
                    "sample_index": list(range(10)),
                    "count_sampled": list(range(5, 15)),
                }
            ),
            pl.DataFrame(
                {
                    "date": date(1, 1, 1),
                    "fd_offset": [0] * 10,
                    "division": [0] * 10,
                    "lineage": ["B"] * 10,
                    "count": [5] * 10,
                    "sample_index": list(range(10)),
                    "count_sampled": list(range(8, 18)),
                }
            ),
        ]
    )

    evaluator = eval.CountsEvaluator(samples=None, data=None, all_counts=df)

    assert evaluator.uncovered_proportion(alpha=0.1) == 0.5
