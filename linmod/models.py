import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl

from .utils import expand_grid, pl_softmax


class HierarchicalDivisionsModel:
    """
    Multinomial regression model with information sharing over divisions.

    Observations are counts of lineages for each division-day.
    See https://doi.org/10.1101/2023.01.02.23284123
    No parameters are constrained here, so specific coefficients are not identifiable.

    counts:         A matrix of counts with shape (num_observations, num_lineages).
    divisions:      A vector of indices representing the division of each observation.
    time:           A vector of the time covariate for each observation.
    """

    def __init__(
        self,
        counts: np.ndarray,
        divisions: np.ndarray,
        time: np.ndarray,
    ):
        self.counts = counts
        self.divisions = divisions
        self.time = time

        self.num_lineages = counts.shape[1]
        self.num_divisions = np.unique(divisions).size

    def numpyro_model(self):
        # beta_0[g, l] is the intercept for lineage l in division g
        mu_beta_0 = numpyro.sample(
            "mu_beta_0",
            dist.StudentT(3, -5, 5),
            sample_shape=(self.num_lineages,),
        )
        sigma_beta_0 = numpyro.sample(
            "sigma_beta_0",
            dist.TruncatedNormal(2, 1, low=0),
            sample_shape=(self.num_lineages,),
        )
        z_0 = numpyro.sample(
            "z_0",
            dist.StudentT(2),
            sample_shape=(self.num_divisions, self.num_lineages),
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
            sample_shape=(self.num_lineages,),
        )
        mu_beta_1 = numpyro.deterministic(
            "mu_beta_1",
            mu_hierarchical + sigma_hierarchical * z_mu,
        )

        # beta_1[g, l] is the slope for lineage l in division g
        sigma_beta_1 = numpyro.sample(
            "sigma_beta_1",
            dist.TruncatedNormal(0.5, 2, low=0),
            sample_shape=(self.num_lineages,),
        )
        Omega_decomposition = numpyro.sample(
            "Omega_decomposition",
            dist.LKJCholesky(self.num_lineages, 2),
        )
        # A faster version of `np.diag(sigma_beta_1) @ Omega_decomposition`
        Sigma_decomposition = sigma_beta_1[:, None] * Omega_decomposition
        z_1 = numpyro.sample(
            "z_1",
            dist.Normal(0, 1),
            sample_shape=(self.num_divisions, self.num_lineages),
        )
        beta_1 = numpyro.deterministic(
            "beta_1",
            mu_beta_1 + z_1 @ Sigma_decomposition.T,
        )

        # z[i, l] is the unnormalized probability of lineage l for observation i
        z = (
            beta_0[self.divisions, :]
            + beta_1[self.divisions, :] * self.time[:, None]
        )

        numpyro.sample(
            "Y",
            dist.Multinomial(total_count=self.counts.sum(axis=1), logits=z),
            obs=self.counts,
        )


class IndependentDivisionsModel:
    """
    Multinomial regression model assuming independence between divisions and specifying
    an intercept and slope on time for each division-lineage.

    Observations are counts of lineages for each division-day.
    See https://doi.org/10.1101/2023.01.02.23284123
    No parameters are constrained here, so specific coefficients are not identifiable.

    divisions:      A Series of division names for each observation.
    time:           A Series of the time covariate for each observation.
    counts:         A DataFrame of counts with shape (num_observations, num_lineages).
                    Column names should correspond to lineage names.
                    Set to `None` to use as a generative model.
    N:              A Series of total counts (across lineages) for each observation.
                    Not required if providing observed `counts`.
    num_lineages:   The number of lineages.
                    Not required if providing observed `counts`.
    """

    def __init__(
        self,
        divisions: pl.Series,
        time: pl.Series,
        counts: pl.DataFrame | None = None,
        N: pl.Series | None = None,
        num_lineages: int | None = None,
    ):
        if counts is None:
            assert num_lineages is not None and N is not None
            self.N = N.to_numpy()
            self.num_lineages = num_lineages
        else:
            assert num_lineages is None and N is None
            self.N = counts.sum_horizontal().to_numpy()
            self.num_lineages = counts.shape[1]

        self._time_standardizer = lambda t: (t - time.mean()) / time.std()

        self.division_names, self.divisions = np.unique(
            divisions, return_inverse=True
        )
        self.time = self._time_standardizer(time.to_numpy())
        self.counts = counts.to_numpy()
        self.lineage_names = counts.columns if counts is not None else None

    def numpyro_model(self):
        with numpyro.plate_stack(
            "division-lineage",
            (np.unique(self.divisions).size, self.num_lineages),
        ):
            with numpyro.handlers.reparam(
                config={
                    "beta_0": numpyro.infer.reparam.LocScaleReparam(
                        centered=0
                    ),
                    "beta_1": numpyro.infer.reparam.LocScaleReparam(
                        centered=0
                    ),
                }
            ):
                # beta_0[g, l] is the intercept for lineage l in division g
                beta_0 = numpyro.sample("beta_0", dist.Normal(0, 1))

                # beta_1[g, l] is the slope for lineage l in division g
                beta_1 = numpyro.sample("beta_1", dist.Normal(0, 1))

        likelihood = multinomial_likelihood(
            beta_0, beta_1, self.divisions, self.time, self.N
        )

        # Y[i, l] is the count of lineage l for observation i
        numpyro.sample("Y", likelihood, obs=self.counts)

    def create_forecasts(self, mcmc, fd_offsets) -> pl.DataFrame:
        parameter_samples = (
            expand_grid(
                chain=np.arange(mcmc.num_chains),
                iteration=np.arange(mcmc.num_samples),
                division=self.division_names,
                lineage=self.lineage_names,
            )
            .with_columns(
                beta_0=np.asarray(mcmc.get_samples()["beta_0"]).flatten(),
                beta_1=np.asarray(mcmc.get_samples()["beta_1"]).flatten(),
                sample_index=pl.col("iteration")
                + pl.col("chain") * mcmc.num_samples
                + 1,
            )
            .drop("chain", "iteration")
        )

        return (
            expand_grid(
                sample_index=parameter_samples["sample_index"].unique(),
                fd_offset=fd_offsets,
            )
            .join(parameter_samples, on="sample_index")
            .with_columns(
                phi=pl_softmax(
                    pl.col("beta_0")
                    + pl.col("beta_1")
                    * self._time_standardizer(pl.col("fd_offset")),
                ).over("sample_index", "division", "fd_offset")
            )
            .drop("beta_0", "beta_1")
        )


class BaselineModel:
    """
    Multinomial model assuming independence between divisions and specifying a
    constant proportion over time for each division-lineage.

    Observations are counts of lineages for each division-day.
    See https://doi.org/10.1101/2023.01.02.23284123
    No parameters are constrained here, so specific coefficients are not identifiable.

    divisions:      A Series of division names for each observation.
    counts:         A DataFrame of counts with shape (num_observations, num_lineages).
                    Column names should correspond to lineage names.
                    Set to `None` to use as a generative model.
    N:              A Series of total counts (across lineages) for each observation.
                    Not required if providing observed `counts`.
    num_lineages:   The number of lineages.
                    Not required if providing observed `counts`.
    """

    def __init__(
        self,
        divisions: pl.Series,
        counts: pl.DataFrame | None = None,
        N: pl.Series | None = None,
        num_lineages: int | None = None,
    ):
        if counts is None:
            assert num_lineages is not None and N is not None
            self.N = N.to_numpy()
            self.num_lineages = num_lineages
        else:
            assert num_lineages is None and N is None
            self.N = counts.sum_horizontal().to_numpy()
            self.num_lineages = counts.shape[1]

        self.division_names, self.divisions = np.unique(
            divisions, return_inverse=True
        )
        self.counts = counts.to_numpy()
        self.lineage_names = counts.columns if counts is not None else None

    def numpyro_model(self):
        with numpyro.plate_stack(
            "division-lineage",
            (np.unique(self.divisions).size, self.num_lineages),
        ):
            with numpyro.handlers.reparam(
                config={
                    "logit_phi": numpyro.infer.reparam.LocScaleReparam(
                        centered=0
                    ),
                }
            ):
                # logit_phi[g, l] is for lineage l in division g
                logit_phi = numpyro.sample("logit_phi", dist.Normal(0, 1))

        likelihood = multinomial_likelihood(
            logit_phi,
            0 * logit_phi,
            self.divisions,
            0 * self.divisions,
            self.N,
        )

        # Y[i, l] is the count of lineage l for observation i
        numpyro.sample("Y", likelihood, obs=self.counts)

    def create_forecasts(self, mcmc, fd_offsets) -> pl.DataFrame:
        parameter_samples = (
            expand_grid(
                chain=np.arange(mcmc.num_chains),
                iteration=np.arange(mcmc.num_samples),
                division=self.division_names,
                lineage=self.lineage_names,
            )
            .with_columns(
                logit_phi=np.asarray(
                    mcmc.get_samples()["logit_phi"]
                ).flatten(),
                sample_index=pl.col("iteration")
                + pl.col("chain") * mcmc.num_samples
                + 1,
            )
            .drop("chain", "iteration")
        )

        return (
            expand_grid(
                sample_index=parameter_samples["sample_index"].unique(),
                fd_offset=fd_offsets,
            )
            .join(parameter_samples, on="sample_index")
            .with_columns(
                phi=pl_softmax(pl.col("logit_phi")).over(
                    "sample_index", "division", "fd_offset"
                )
            )
            .drop("logit_phi")
        )


def multinomial_likelihood(
    beta_0: np.ndarray,
    beta_1: np.ndarray,
    divisions: np.ndarray,
    time: np.ndarray,
    N: np.ndarray | None = None,
):
    """
    Distribution of observations for multinomial regression model for
    a single human population. Observations are counts of lineages for each
    time.

    beta_0 (np.ndarray):        Intercept, shape (num_divisions, num_lineages)
    beta_1 (np.ndarray):        Slope on time, shape (num_divisions, num_lineages)
    divisions (np.ndarray):     Division index for each observation,
                                length (num_observations)
    time (np.ndarray):          Times, length (num_observations)
    N (np.ndarray):             Total counts across lineages, length (num_observations)
    """

    # Shape checks
    assert beta_0.shape == beta_1.shape
    assert time.shape == divisions.shape == N.shape

    # z[i, l] is the unnormalized probability of lineage l for observation i
    z = beta_0[divisions, :] + beta_1[divisions, :] * time[:, None]
    assert z.shape == (N.shape[0], beta_0.shape[1])

    return dist.Multinomial(total_count=N, logits=z)
