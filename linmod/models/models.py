import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl

from ..utils import expand_grid, pl_softmax


class HierarchicalDivisionsModel:
    """
    Multinomial regression model with information sharing over divisions.

    Observations are counts of lineages for each division-day.
    See https://doi.org/10.1101/2023.01.02.23284123
    No parameters are constrained here, so specific coefficients are not identifiable.

    data:           A DataFrame with the standard model input format.
    N:              A Series of total counts (across lineages) for each observation.
                    Only required if a generative model is desired; lineage counts will
                    then be ignored.
    num_lineages:   The number of lineages. Only required if a generative model is
                    desired; lineage counts will then be ignored.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        N: pl.Series | None = None,
        num_lineages: int | None = None,
        num_divisions: int | None = None,
    ):
        data = data.pivot(
            on="lineage", index=["fd_offset", "division"], values="count"
        ).fill_null(0)

        self.lineage_names = sorted(
            filter(lambda c: c not in ["fd_offset", "division"], data.columns)
        )
        self.division_names, self.divisions = np.unique(
            data["division"], return_inverse=True
        )
        self.counts = data.select(self.lineage_names).to_numpy()

        time = data["fd_offset"].to_numpy()
        self._time_standardizer = lambda t: (t - time.mean()) / time.std()
        self.time = self._time_standardizer(time)

        if (
            num_lineages is not None
            and N is not None
            and num_divisions is not None
        ):
            self.N = N.to_numpy()
            self.num_lineages = num_lineages
            self.num_divisions = num_divisions
            self.counts = None
        else:
            assert (
                num_lineages is None and N is None and num_divisions is None
            ), "To use as a generative model, supply both `num_lineages` and `N`."

            self.N = self.counts.sum(axis=1)
            self.num_lineages = self.counts.shape[1]
            self.num_divisions = len(self.division_names)

    def numpyro_model(self):
        # beta_0[g, l] is the intercept for lineage l in division g
        mu_beta_0 = numpyro.sample(
            "mu_beta_0",
            dist.Normal(0, 0.70710678),
            sample_shape=(self.num_lineages,),
        )
        sigma_beta_0 = numpyro.sample(
            "sigma_beta_0",
            dist.Exponential(1.0),
            sample_shape=(self.num_lineages,),
        )
        z_0 = numpyro.sample(
            "z_0",
            dist.Normal(0, 0.70710678),
            sample_shape=(self.num_divisions, self.num_lineages),
        )
        beta_0 = numpyro.deterministic(
            "beta_0",
            mu_beta_0 + sigma_beta_0 * z_0,
        )

        # mu_beta_1[l] is the mean of the slope for lineage l
        mu_beta_1 = numpyro.sample(
            "mu_beta_1",
            dist.Normal(0, 0.1767767),
            sample_shape=(self.num_lineages,),
        )

        # beta_1[g, l] is the slope for lineage l in division g
        sigma_beta_1 = 0.1767767 * np.ones(self.num_lineages)
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


class IndependentDivisionsModel:
    """
    Multinomial regression model assuming independence between divisions and specifying
    an intercept and slope on time for each division-lineage.

    Observations are counts of lineages for each division-day.
    See https://doi.org/10.1101/2023.01.02.23284123
    No parameters are constrained here, so specific coefficients are not identifiable.

    data:           A DataFrame with the standard model input format.
    N:              A Series of total counts (across lineages) for each observation.
                    Only required if a generative model is desired; lineage counts will
                    then be ignored.
    num_lineages:   The number of lineages. Only required if a generative model is
                    desired; lineage counts will then be ignored.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        N: pl.Series | None = None,
        num_lineages: int | None = None,
    ):
        data = data.pivot(
            on="lineage", index=["fd_offset", "division"], values="count"
        ).fill_null(0)

        self.lineage_names = sorted(
            filter(lambda c: c not in ["fd_offset", "division"], data.columns)
        )
        self.division_names, self.divisions = np.unique(
            data["division"], return_inverse=True
        )
        self.counts = data.select(self.lineage_names).to_numpy()

        time = data["fd_offset"].to_numpy()
        self._time_standardizer = lambda t: (t - time.mean()) / time.std()
        self.time = self._time_standardizer(time)

        if num_lineages is not None and N is not None:
            self.N = N.to_numpy()
            self.num_lineages = num_lineages
            self.counts = None
        else:
            assert (
                num_lineages is None and N is None
            ), "To use as a generative model, supply both `num_lineages` and `N`."

            self.N = self.counts.sum(axis=1)
            self.num_lineages = self.counts.shape[1]

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
                beta_1 = numpyro.sample("beta_1", dist.Normal(0, 0.25))

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

    data:           A DataFrame with the standard model input format.
    N:              A Series of total counts (across lineages) for each observation.
                    Only required if a generative model is desired; lineage counts will
                    then be ignored.
    num_lineages:   The number of lineages. Only required if a generative model is
                    desired; lineage counts will then be ignored.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        N: pl.Series | None = None,
        num_lineages: int | None = None,
    ):
        data = data.pivot(
            on="lineage", index=["fd_offset", "division"], values="count"
        ).fill_null(0)

        self.lineage_names = sorted(
            filter(lambda c: c not in ["fd_offset", "division"], data.columns)
        )
        self.division_names, self.divisions = np.unique(
            data["division"], return_inverse=True
        )
        self.counts = data.select(self.lineage_names).to_numpy()

        if num_lineages is not None and N is not None:
            self.N = N.to_numpy()
            self.num_lineages = num_lineages
            self.counts = None
        else:
            assert (
                num_lineages is None and N is None
            ), "To use as a generative model, supply both `num_lineages` and `N`."

            self.N = self.counts.sum(axis=1)
            self.num_lineages = self.counts.shape[1]

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
