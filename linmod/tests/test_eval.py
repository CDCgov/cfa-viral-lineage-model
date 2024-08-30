import numpy as np
from numpy.random import default_rng

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

    data = data.with_columns(count=rng.integers(0, 100, data.shape[0]))

    # Generate fake forecasts of population proportions
    samples = eval._merge_samples_and_data(
        data,
        expand_grid(
            fd_offset=range(num_days),
            division=range(num_divisions),
            lineage=range(num_lineages),
            sample_index=range(num_samples),
        ),
        samples_are_phi=True,
    ).rename({"phi": "phi_mean"})

    samples = samples.with_columns(
        phi=rng.normal(samples["phi_mean"], np.sqrt(sample_variance))
    ).drop("phi_mean")

    return samples.lazy(), data.lazy()


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
        eval.score(
            data,
            eval.mean_norm_per_division_day,
            samples,
            samples_are_phi=True,
            p=1,
        ),
        np.sqrt(sample_variance * 2 / np.pi)
        * NUM_DAYS
        * NUM_DIVISIONS
        * NUM_LINEAGES,
        atol=atol,
    )


# def test_proportions_mean_L1_norm2():
#     samples, data = _generate_fake_samples_and_data(
#         None,
#         num_days=2,
#         num_divisions=2,
#         num_lineages=2,
#         num_samples=2,
#         sample_variance=0,
#     )

#     # Put in fixed values for counts and phi
#     data = data.sort("fd_offset", "division", "lineage").with_columns(
#         count=pl.Series([1, 2, 3, 4, 5, 6, 7, 8]),
#     )

#     samples = samples.sort(
#         "fd_offset", "division", "sample_index", "lineage"
#     ).with_columns(
#         phi=pl.Series(
#             [
#                 0.2,
#                 0.8,
#                 0.3,
#                 0.7,
#                 0.4,
#                 0.6,
#                 0.5,
#                 0.5,
#                 0.2,
#                 0.8,
#                 0.3,
#                 0.7,
#                 0.4,
#                 0.6,
#                 0.5,
#                 0.5,
#             ]
#         ),
#     )

#     correct_output = pl.DataFrame(
#         {
#             "fd_offset": [0, 0, 1, 1],
#             "division": [0, 1, 0, 1],
#             "mean_norm": [
#                 np.abs(np.array([0.2, 0.3]) - 1 / (1 + 2)).mean()
#                 + np.abs(np.array([0.8, 0.7]) - 2 / (1 + 2)).mean(),
#                 np.abs(np.array([0.4, 0.5]) - 3 / (3 + 4)).mean()
#                 + np.abs(np.array([0.6, 0.5]) - 4 / (3 + 4)).mean(),
#                 np.abs(np.array([0.2, 0.3]) - 5 / (5 + 6)).mean()
#                 + np.abs(np.array([0.8, 0.7]) - 6 / (5 + 6)).mean(),
#                 np.abs(np.array([0.4, 0.5]) - 7 / (7 + 8)).mean()
#                 + np.abs(np.array([0.6, 0.5]) - 8 / (7 + 8)).mean(),
#             ],
#         }
#     )

#     result = eval.mean_norm_per_division_day(
#         data, samples, samples_are_phi=True, p=1
#     ).collect()

#     assert_frame_equal(
#         result,
#         correct_output,
#         check_row_order=False,
#         check_column_order=True,
#         rtol=0,
#         atol=0,
#     )


# def test_proportions_L1_energy_score(
#     num_samples=100000, sample_variance=1.4, atol=0.05
# ):
#     r"""
#     Test the estimate of the energy score of phi.

#     The setup is as follows:
#     - Generate fake data $Y_{tgl} \sim Uniform{0, ..., 99}$
#     - Denote true proportion as $\phi_{tgl} = \frac{Y_{tgl}}{Y_{tg \cdot}}$
#     - Generate fake "forecasts" $f_{tgl} \sim N(\phi_{tgl}, \sigma^2)$
#     - Check $$
#         \sum_{t, g} E[ || f_{tg} - \phi_{tg} ||_2 ]
#         - \frac{1}{2} E[ || f_{tg} - f_{tg}' ||_2 ]
#       $$ as reported by `eval.proportions_energy_score`

#     Note that our "forecasts" are not actually proportions, but independent normal
#     random variables. This is so that the quantity can be computed analytically.
#     """

#     rng = default_rng()

#     NUM_DAYS = 4
#     NUM_DIVISIONS = 3
#     NUM_LINEAGES = 2

#     samples, data = _generate_fake_samples_and_data(
#         rng,
#         num_days=NUM_DAYS,
#         num_divisions=NUM_DIVISIONS,
#         num_lineages=NUM_LINEAGES,
#         num_samples=num_samples,
#         sample_variance=sample_variance,
#     )

#     # Because we use L1 norm, term 1 is equal to the sum of each component's MAE from
#     # its mean.
#     # The mean absolute error of a normal random variable from its mean is
#     # $\sigma \sqrt{\frac{2}{\pi}}$ (because $X - \mu \sim N(0, \sigma^2)$, so
#     # $|X - \mu| \sim \text{half-normal}(\sigma)$, which has this mean).

#     term1 = (
#         np.sqrt(2 * sample_variance / np.pi)
#         * NUM_DAYS
#         * NUM_DIVISIONS
#         * NUM_LINEAGES
#     )

#     # Because we use L1 norm, term 2 is equal to the sum of each component's MAE from
#     # an independent copy of itself.
#     # The mean absolute error of a normal random variable from an independent copy is
#     # $\frac{2\sigma}{\sqrt{\pi}}$ (because $X - X' \sim N(0, 2 \sigma^2)$, so
#     # $|X - X'| \sim \text{half-normal}(\sqrt{2} \sigma)$, which has this mean).

#     term2 = (
#         (2 * np.sqrt(sample_variance / np.pi))
#         * NUM_DAYS
#         * NUM_DIVISIONS
#         * NUM_LINEAGES
#     )

#     assert np.isclose(
#         eval.score(data, eval.energy_score_per_division_day, samples, samples_are_phi=True, p=1),
#         term1 - 0.5 * term2,
#         atol=atol,
#     )


# def test_proportions_L1_energy_score2():
#     samples, data = _generate_fake_samples_and_data(
#         None,
#         num_days=2,
#         num_divisions=2,
#         num_lineages=2,
#         num_samples=3,
#         sample_variance=0,
#     )

#     # Put in fixed values for counts and phi
#     data = data.sort("fd_offset", "division", "lineage").with_columns(
#         count=pl.Series([1, 2, 3, 4, 5, 6, 7, 8]),
#     )

#     samples = samples.sort(
#         "fd_offset", "division", "sample_index", "lineage"
#     ).with_columns(
#         phi=pl.Series(
#             [
#                 0.2,
#                 0.8,
#                 0.3,
#                 0.7,
#                 0.1,
#                 0.9,
#                 0.4,
#                 0.6,
#                 0.5,
#                 0.5,
#                 0.1,
#                 0.9,
#                 0.2,
#                 0.8,
#                 0.3,
#                 0.7,
#                 0.1,
#                 0.9,
#                 0.4,
#                 0.6,
#                 0.5,
#                 0.5,
#                 0.1,
#                 0.9,
#             ]
#         ),
#     )

#     correct_output = pl.DataFrame(
#         {
#             "fd_offset": [0, 0, 1, 1],
#             "division": [0, 1, 0, 1],
#             "energy_score": [
#                 np.abs(np.array([0.2, 0.3, 0.1]) - 1 / (1 + 2)).mean()
#                 + np.abs(np.array([0.8, 0.7, 0.9]) - 2 / (1 + 2)).mean()
#                 - 0.5
#                 * np.abs(
#                     np.array([0.2, 0.3, 0.1]) - np.array([0.1, 0.2, 0.3])
#                 ).mean()
#                 - 0.5
#                 * np.abs(
#                     np.array([0.8, 0.7, 0.9]) - np.array([0.9, 0.8, 0.7])
#                 ).mean(),
#                 np.abs(np.array([0.4, 0.5, 0.1]) - 3 / (3 + 4)).mean()
#                 + np.abs(np.array([0.6, 0.5, 0.9]) - 4 / (3 + 4)).mean()
#                 - 0.5
#                 * np.abs(
#                     np.array([0.4, 0.5, 0.1]) - np.array([0.1, 0.4, 0.5])
#                 ).mean()
#                 - 0.5
#                 * np.abs(
#                     np.array([0.6, 0.5, 0.9]) - np.array([0.9, 0.6, 0.5])
#                 ).mean(),
#                 np.abs(np.array([0.2, 0.3, 0.1]) - 5 / (5 + 6)).mean()
#                 + np.abs(np.array([0.8, 0.7, 0.9]) - 6 / (5 + 6)).mean()
#                 - 0.5
#                 * np.abs(
#                     np.array([0.2, 0.3, 0.1]) - np.array([0.1, 0.2, 0.3])
#                 ).mean()
#                 - 0.5
#                 * np.abs(
#                     np.array([0.8, 0.7, 0.9]) - np.array([0.9, 0.8, 0.7])
#                 ).mean(),
#                 np.abs(np.array([0.4, 0.5, 0.1]) - 7 / (7 + 8)).mean()
#                 + np.abs(np.array([0.6, 0.5, 0.9]) - 8 / (7 + 8)).mean()
#                 - 0.5
#                 * np.abs(
#                     np.array([0.4, 0.5, 0.1]) - np.array([0.1, 0.4, 0.5])
#                 ).mean()
#                 - 0.5
#                 * np.abs(
#                     np.array([0.6, 0.5, 0.9]) - np.array([0.9, 0.6, 0.5])
#                 ).mean(),
#             ],
#         }
#     )

#     result = eval.energy_score_per_division_day(
#         data, samples, samples_are_phi=True, p=1
#     ).collect()

#     assert_frame_equal(
#         result,
#         correct_output,
#         check_row_order=False,
#         check_column_order=True,
#     )
