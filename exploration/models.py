import numpy as np
import numpyro
import numpyro.distributions as dist


def hierarchical_divisions_model(
    counts: np.ndarray,
    divisions: np.ndarray,
    time: np.ndarray,
):
    """
    Multinomial regression model with information sharing over divisions.
    Observations are counts of lineages for each division-day.
    See https://doi.org/10.1101/2023.01.02.23284123
    No parameters are constrained here, so specific coefficients are not identifiable.

    counts:         A matrix of counts with shape (num_observations, num_lineages).
    divisions:      A vector of indices representing the division of each observation.
    time:           A vector of the time covariate for each observation.
    """

    num_lineages = counts.shape[1]
    num_divisions = np.unique(divisions).size

    # beta_0[g, l] is the intercept for lineage l in division g
    mu_beta_0 = numpyro.sample(
        "mu_beta_0",
        dist.StudentT(3, -5, 5),
        sample_shape=(num_lineages,),
    )
    sigma_beta_0 = numpyro.sample(
        "sigma_beta_0",
        dist.TruncatedNormal(2, 1, low=0),
        sample_shape=(num_lineages,),
    )
    z_0 = numpyro.sample(
        "z_0",
        dist.StudentT(2),
        sample_shape=(num_divisions, num_lineages),
    )
    beta_0 = numpyro.deterministic(
        "beta_0",
        mu_beta_0 + sigma_beta_0 * z_0,
    )

    # mu_beta_1[l] is the mean of the slope for lineage l
    mu_hierarchical = numpyro.sample(
        "mu_hierarchical",
        dist.Normal(-1, np.sqrt(0.5)),
    )
    sigma_hierarchical = numpyro.sample(
        "sigma_hierarchical",
        dist.TruncatedNormal(1, np.sqrt(0.1), low=0),
    )
    z_mu = numpyro.sample(
        "z_mu",
        dist.Normal(0, 1),
        sample_shape=(num_lineages,),
    )
    mu_beta_1 = numpyro.deterministic(
        "mu_beta_1",
        mu_hierarchical + sigma_hierarchical * z_mu,
    )

    # beta_1[g, l] is the slope for lineage l in division g
    sigma_beta_1 = numpyro.sample(
        "sigma_beta_1",
        dist.TruncatedNormal(0.5, 2, low=0),
        sample_shape=(num_lineages,),
    )
    Omega_decomposition = numpyro.sample(
        "Omega_decomposition",
        dist.LKJCholesky(num_lineages, 2),
    )
    # A faster version of `np.diag(sigma_beta_1) @ Omega_decomposition`
    Sigma_decomposition = sigma_beta_1[:, None] * Omega_decomposition
    z_1 = numpyro.sample(
        "z_1",
        dist.Normal(0, 1),
        sample_shape=(num_divisions, num_lineages),
    )
    beta_1 = numpyro.deterministic(
        "beta_1",
        mu_beta_1 + z_1 @ Sigma_decomposition.T,
    )

    # z[i, l] is the unnormalized probability of lineage l for observation i
    z = beta_0[divisions, :] + beta_1[divisions, :] * time[:, None]

    numpyro.sample(
        "Y",
        dist.Multinomial(total_count=counts.sum(axis=1), logits=z),
        obs=counts,
    )


def independent_divisions_model(
    divisions: np.ndarray,
    time: np.ndarray,
    counts: np.ndarray | None = None,
    N: np.ndarray | None = None,
    num_lineages: int | None = None,
):
    """
    Multinomial regression model assuming independence between divisions.
    Observations are counts of lineages for each division-day.
    See https://doi.org/10.1101/2023.01.02.23284123
    No parameters are constrained here, so specific coefficients are not identifiable.

    divisions:      A vector of indices representing the division of each observation.
    time:           A vector of the time covariate for each observation.
    counts:         A matrix of counts with shape (num_observations, num_lineages).
                    Set to `None` to use as a generative model.
    N:              A vector of total counts (across lineages) for each observation.
                    Not required if providing observed `counts`.
    num_lineages:   The number of lineages.
                    Not required if providing observed `counts`.

    """

    if counts is None:
        assert num_lineages is not None and N is not None
    else:
        assert num_lineages is None and N is None

        N = counts.sum(axis=1)
        num_lineages = counts.shape[1]

    with numpyro.plate_stack(
        "division-lineage", (np.unique(divisions).size, num_lineages)
    ):
        with numpyro.handlers.reparam(
            config={
                "beta_0": numpyro.infer.reparam.LocScaleReparam(
                    centered=0, shape_params=("df",)
                ),
                "beta_1": numpyro.infer.reparam.LocScaleReparam(centered=0),
            }
        ):
            # beta_0[g, l] is the intercept for lineage l in division g
            beta_0 = numpyro.sample(
                "beta_0",
                dist.StudentT(2, loc=-5, scale=2),
            )

            # beta_1[g, l] is the slope for lineage l in division g
            beta_1 = numpyro.sample(
                "beta_1",
                dist.Normal(-1, 1.8),
            )

    # z[i, l] is the unnormalized probability of lineage l for observation i
    z = beta_0[divisions, :] + beta_1[divisions, :] * time[:, None]

    numpyro.sample(
        "Y",
        dist.Multinomial(total_count=N, logits=z),
        obs=counts,
    )


def one_division_dist(
    beta_0: np.ndarray,
    beta_1: np.ndarray,
    time: np.ndarray,
    N: np.ndarray | None = None,
):
    """
    Distribution of observations for multinomial regression model for
    a single human population. Observations are counts of lineages for each
    time.

    beta_0 (np.ndarray): "intercept", one per lineage
    beta_1 (np.ndarray): "slope", one per lineage
    time (np.ndarray): times, one per observation
    N (np.ndarray): total counts across lineages, one per observation
    """
    # all per-lineage parameters should have same length
    num_lineages = len(beta_0)
    assert len(beta_1) == num_lineages

    # all per-observation parameters should have same length
    num_obs = len(time)
    assert len(N) == num_obs

    # logits should have shape (lineages, observations)
    z = beta_0[:, np.newaxis] + np.outer(beta_1, time)
    assert z.shape == (num_lineages, num_obs)

    return dist.Multinomial(total_count=N, logits=z)


def one_division_sample(
    time: np.ndarray, counts: np.ndarray | None = None, var: str = "Y"
):
    """
    Multinomial regression model for a single human population.
    Observations are counts of lineages for each division-day.
    See https://doi.org/10.1101/2023.01.02.23284123
    No parameters are constrained here, so specific coefficients are not identifiable.

    time:           A vector of the time covariate for each observation.
    counts:         A matrix of counts with shape (num_observations, num_lineages).
    var (str): Variable name. Default: "Y"
    """

    N = counts.sum(axis=1)
    num_lineages = counts.shape[1]

    with numpyro.plate_stack("lineage", num_lineages):
        with numpyro.handlers.reparam(
            config={
                "beta_0": numpyro.infer.reparam.LocScaleReparam(
                    centered=0, shape_params=("df",)
                ),
                "beta_1": numpyro.infer.reparam.LocScaleReparam(centered=0),
            }
        ):
            # beta_0[g, l] is the intercept for lineage l in division g
            beta_0 = numpyro.sample(
                "beta_0",
                dist.StudentT(2, loc=-5, scale=2),
            )

            # beta_1[g, l] is the slope for lineage l in division g
            beta_1 = numpyro.sample(
                "beta_1",
                dist.Normal(-1, 1.8),
            )

    obs_dist = one_division_dist(beta_0=beta_0, beta_1=beta_1, time=time, N=N)

    return numpyro.sample(var, obs_dist, obs=counts)
