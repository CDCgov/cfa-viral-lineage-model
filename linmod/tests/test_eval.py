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

    # Generate fake population counts
    data = expand_grid(
        fd_offset=range(num_days),
        division=range(num_divisions),
        lineage=range(num_lineages),
    )

    data = data.with_columns(count=rng.integers(0, 100, data.shape[0]))

    # Generate fake forecasts of population proportions
    samples = eval._merge_samples_and_data(
        expand_grid(
            fd_offset=range(num_days),
            division=range(num_divisions),
            lineage=range(num_lineages),
            sample_index=range(num_samples),
        ),
        data,
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
    - Denote true proportion as $\pi_{tgl} = \frac{Y_{tgl}}{Y_{tg \cdot}}$
    - Generate fake "forecasts" $phi_{tgl} \sim N(\pi_{tgl}, \sigma^2)$
    - Check $\sum_{t, g} E[ || phi_{tg} - \pi_{tg} ||_1 ]$
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
    # The MAE of a normal random variable from its mean is $\sigma \sqrt{2/\pi}$.

    assert np.isclose(
        eval.proportions_mean_norm(samples, data, L=1),
        np.sqrt(sample_variance * 2 / np.pi)
        * NUM_DAYS
        * NUM_DIVISIONS
        * NUM_LINEAGES,
        atol=atol,
    )


def test_proportions_L1_energy_score(
    num_samples=100000, sample_variance=1.4, atol=0.05
):
    r"""
    Test the estimate of the energy score of phi.

    The setup is as follows:
    - Generate fake data $Y_{tgl} \sim Uniform{0, ..., 99}$
    - Denote true proportion as $\pi_{tgl} = \frac{Y_{tgl}}{Y_{tg \cdot}}$
    - Generate fake "forecasts" $phi_{tgl} \sim N(\pi_{tgl}, \sigma^2)$
    - Check $\sum_{t, g} E[ || phi_{tg} - \pi_{tg} ||_2 ] - \frac{1}{2} E[ || phi_{tg} - \phi_{tg}' ||_2 ]$
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

    # Because we use L1 norm, term 1 is equal to the sum of each component's MAE from
    # its mean.
    # The MAE of a normal random variable from its mean is $\sigma \sqrt{\frac{2}{\pi}}$.

    term1 = (
        np.sqrt(2 * sample_variance / np.pi)
        * NUM_DAYS
        * NUM_DIVISIONS
        * NUM_LINEAGES
    )

    # Because we use L1 norm, term 2 is equal to the sum of each component's MAE from
    # an independent copy of itself.
    # The MAE of a normal random variable from an independent copy is $\frac{2\sigma}{\sqrt{\pi}}$.

    term2 = (
        (2 * np.sqrt(sample_variance / np.pi))
        * NUM_DAYS
        * NUM_DIVISIONS
        * NUM_LINEAGES
    )

    assert np.isclose(
        eval.proportions_energy_score(samples, data, L=1),
        term1 - 0.5 * term2,
        atol=atol,
    )


test_proportions_mean_L1_norm()
test_proportions_L1_energy_score()
