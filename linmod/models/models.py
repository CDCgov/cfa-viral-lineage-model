from abc import ABC, abstractmethod
from collections.abc import Sequence

import dendropy
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl

from ..utils import expand_grid, pl_softmax
from .utils import ForecastFrame


class MultinomialModel(ABC):
    def dense_mass(self):
        """
        For use by numpyro.infer.NUTS, specification of structure of mass matrix.

        Defaults to diagonal mass matrix.
        """
        return False

    def ignore_nan_in(self):
        """
        For use by get_convergence, list of model "sites" where NaN convergence
        results are expected, such as a constant value.

        Defaults to assuming NaNs are unexpected.
        """
        return []

    @abstractmethod
    def numpyro_model(self):
        """
        A NumPyro model suitable for use as `model` argument to numpyro.infer.NUTS.
        """
        raise NotImplementedError

    @abstractmethod
    def create_forecasts(self, mcmc, fd_offsets) -> ForecastFrame:
        """
        Generate a data frame of forecasted population-level proportions.

        mcmc:          The MCMC.
        fd_offsets:    The (relative) days on which to generate forecasted proportions.
        """
        raise NotImplementedError


class HierarchicalDivisionsModel(MultinomialModel):
    """
    Multinomial regression model with information sharing over divisions.

    Observations are counts of lineages for each division-day.
    No parameters are constrained here, so specific coefficients are not identifiable.

    data:              A DataFrame with the standard model input format.
    N:                 A Series of total counts (across lineages) for each observation.
                       Only required if a generative model is desired; lineage counts will
                       then be ignored.
    num_lineages:      The number of lineages. Only required if a generative model is
                       desired; lineage counts will then be ignored.
    pool_intercepts    A float in [0,1] which determines how strongly the intercepts are pooled.
                       Determines the proportion of prior variance on per-division intercepts which
                       comes from the prior variance on the shared hierarchical mean.
    pool_slopes        Equivalent of `pool_intercepts` for slopes.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        N: pl.Series | None = None,
        num_lineages: int | None = None,
        num_divisions: int | None = None,
        pool_intercepts: float = 0.5,
        pool_slopes: float = 0.75,
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

        self.time = data["fd_offset"].to_numpy()

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

        # Strength of pooling, closer to 1 means more sharing across divisions
        assert (pool_slopes >= 0.0 and pool_slopes <= 1.0) and (
            pool_intercepts >= 0.0 and pool_intercepts <= 1.0
        ), "Pooling strengths must be in [0,1]"
        self.pool_beta_0 = pool_intercepts
        self.pool_beta_1 = pool_slopes

    def dense_mass(self):
        block_diag_int = [
            (f"z_0_{grp}",) for grp in range(np.unique(self.divisions).size)
        ]
        block_diag_slope = [
            (f"z_1_{grp}",) for grp in range(np.unique(self.divisions).size)
        ]
        return [
            ("z_mu_beta_0",),
            ("z_mu_beta_1",),
            *block_diag_slope,
            *block_diag_int,
        ]

    def numpyro_model(self):
        sigma_beta_0 = 1.0
        sigma_beta_1 = 0.25

        sigma_global_beta_0 = jnp.sqrt(self.pool_beta_0 * sigma_beta_0)
        sigma_global_beta_1 = jnp.sqrt(self.pool_beta_1 * sigma_beta_1)

        sigma_local_beta_0 = jnp.sqrt((1.0 - self.pool_beta_0) * sigma_beta_0)
        sigma_local_beta_1 = jnp.sqrt((1.0 - self.pool_beta_1) * sigma_beta_1)

        z_mu_beta_0 = numpyro.sample(
            "z_mu_beta_0", dist.Normal(0, 1), sample_shape=(self.num_lineages,)
        )

        z_mu_beta_1 = numpyro.sample(
            "z_mu_beta_1", dist.Normal(0, 1), sample_shape=(self.num_lineages,)
        )

        mu_beta_0 = numpyro.deterministic(
            "mu_beta_0", z_mu_beta_0 * sigma_global_beta_0
        )
        mu_beta_1 = numpyro.deterministic(
            "mu_beta_1", z_mu_beta_1 * sigma_global_beta_1
        )

        v_z_0 = []
        v_z_1 = []

        for i in range(np.unique(self.divisions).size):
            z_0 = numpyro.sample(
                f"z_0_{i}",
                dist.Normal(0.0, 1.0),
                sample_shape=(self.num_lineages,),
            )

            z_1 = numpyro.sample(
                f"z_1_{i}",
                dist.Normal(0.0, 1.0),
                sample_shape=(self.num_lineages,),
            )

            v_z_0.append(z_0)
            v_z_1.append(z_1)

        beta_0 = numpyro.deterministic(
            "beta_0",
            mu_beta_0 + (sigma_local_beta_0 * jnp.column_stack(v_z_0)).T,
        )

        beta_1 = numpyro.deterministic(
            "beta_1",
            mu_beta_1 + (sigma_local_beta_1 * jnp.column_stack(v_z_1)).T,
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
                iteration=np.arange(mcmc.num_samples // mcmc.thinning),
                division=self.division_names,
                lineage=self.lineage_names,
            )
            .with_columns(
                beta_0=np.asarray(mcmc.get_samples()["beta_0"]).flatten(),
                beta_1=np.asarray(mcmc.get_samples()["beta_1"]).flatten(),
                sample_index=pl.col("iteration")
                + pl.col("chain") * (mcmc.num_samples // mcmc.thinning)
                + 1,
            )
            .drop("chain", "iteration")
        )

        return ForecastFrame(
            expand_grid(
                sample_index=parameter_samples["sample_index"].unique(),
                fd_offset=fd_offsets,
            )
            .join(parameter_samples, on="sample_index")
            .with_columns(
                phi=pl_softmax(
                    pl.col("beta_0") + pl.col("beta_1") * pl.col("fd_offset"),
                ).over("sample_index", "division", "fd_offset")
            )
            .drop("beta_0", "beta_1")
        )


class PhyloCorrelatedHierarchicalDivisionsModel(HierarchicalDivisionsModel):
    """
    As HierarchicalDivisions but with phylogenetic correlations in the mu_beta_1 terms.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        tree: dendropy.Tree,
        N: pl.Series | None = None,
        num_lineages: int | None = None,
        num_divisions: int | None = None,
        pool_intercepts: float = 0.5,
        pool_slopes: float = 0.75,
    ):
        super().__init__(
            data, N, num_lineages, num_divisions, pool_intercepts, pool_slopes
        )
        self.phylo_vcv = PhyloCorrelatedHierarchicalDivisionsModel.tree_to_vcv(
            tree, self.lineage_names
        )

    @staticmethod
    def tree_to_vcv(tree: dendropy.Tree, lineages: Sequence[str]):
        n_lineages = len(lineages)
        has_other = "other" in lineages
        assert len(tree) == len(lineages) - (1 if has_other else 0)
        _ = np.zeros(
            (
                n_lineages,
                n_lineages,
            )
        )

        # Use tree.node_distance_matrix() to get a distance matrix of nodes
        # Use NDM.distance(leaf node, tree root) to get variances
        # Use NDM.mrca() to get the MRCA of any two taxa, and then NDM.distance(mrca, root) to get covariances
        # >>> dm.distance(tree.leaf_nodes()[0], tree.leaf_nodes()[10], is_weighted_edge_distances=False)
        # 13
        # >>> dm.distance(tree.leaf_nodes()[0], tree.leaf_nodes()[1], is_weighted_edge_distances=False)
        # 2
        # >>> dm.distance(tree.leaf_nodes()[0], tree.seed_node, is_weighted_edge_distances=False)

        # crude but gets the job done
        for leaf_i in tree.leaf_nodes():
            if leaf_i == "other":
                continue

        # normalize to covariance matrix

    def numpyro_model(self):
        # To heat the covariance matrix, use p * CorMat + (1 - p) diag(ones)
        # Apply to: deviations per state (probably?), overall mean (maybe?)
        sigma_beta_0 = 1.0
        sigma_beta_1 = 0.25

        sigma_global_beta_0 = jnp.sqrt(self.pool_beta_0 * sigma_beta_0)
        sigma_global_beta_1 = jnp.sqrt(self.pool_beta_1 * sigma_beta_1)

        sigma_local_beta_0 = jnp.sqrt((1.0 - self.pool_beta_0) * sigma_beta_0)
        sigma_local_beta_1 = jnp.sqrt((1.0 - self.pool_beta_1) * sigma_beta_1)

        z_mu_beta_0 = numpyro.sample(
            "z_mu_beta_0", dist.Normal(0, 1), sample_shape=(self.num_lineages,)
        )

        z_mu_beta_1 = numpyro.sample(
            "z_mu_beta_1", dist.Normal(0, 1), sample_shape=(self.num_lineages,)
        )

        mu_beta_0 = numpyro.deterministic(
            "mu_beta_0", z_mu_beta_0 * sigma_global_beta_0
        )
        mu_beta_1 = numpyro.deterministic(
            "mu_beta_1", z_mu_beta_1 * sigma_global_beta_1
        )

        v_z_0 = []
        v_z_1 = []

        for i in range(np.unique(self.divisions).size):
            z_0 = numpyro.sample(
                f"z_0_{i}",
                dist.Normal(0.0, 1.0),
                sample_shape=(self.num_lineages,),
            )

            z_1 = numpyro.sample(
                f"z_1_{i}",
                dist.Normal(0.0, 1.0),
                sample_shape=(self.num_lineages,),
            )

            v_z_0.append(z_0)
            v_z_1.append(z_1)

        beta_0 = numpyro.deterministic(
            "beta_0",
            mu_beta_0 + (sigma_local_beta_0 * jnp.column_stack(v_z_0)).T,
        )

        beta_1 = numpyro.deterministic(
            "beta_1",
            mu_beta_1 + (sigma_local_beta_1 * jnp.column_stack(v_z_1)).T,
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
                iteration=np.arange(mcmc.num_samples // mcmc.thinning),
                division=self.division_names,
                lineage=self.lineage_names,
            )
            .with_columns(
                beta_0=np.asarray(mcmc.get_samples()["beta_0"]).flatten(),
                beta_1=np.asarray(mcmc.get_samples()["beta_1"]).flatten(),
                sample_index=pl.col("iteration")
                + pl.col("chain") * (mcmc.num_samples // mcmc.thinning)
                + 1,
            )
            .drop("chain", "iteration")
        )

        return ForecastFrame(
            expand_grid(
                sample_index=parameter_samples["sample_index"].unique(),
                fd_offset=fd_offsets,
            )
            .join(parameter_samples, on="sample_index")
            .with_columns(
                phi=pl_softmax(
                    pl.col("beta_0") + pl.col("beta_1") * pl.col("fd_offset"),
                ).over("sample_index", "division", "fd_offset")
            )
            .drop("beta_0", "beta_1")
        )


class CorrelatedDeviationsModel(HierarchicalDivisionsModel):
    """
    An extension of the simple hierarchical model that adds correlations among the lineages
    in how they deviate across sites from the global mean slopes,
    as in https://doi.org/10.1101/2023.01.02.23284123
    """

    def dense_mass(self):
        block_diag_int = [
            (f"z_0_{grp}",) for grp in range(np.unique(self.divisions).size)
        ]
        block_diag_slope = [
            (f"z_1_{grp}",) for grp in range(np.unique(self.divisions).size)
        ]
        return [
            ("z_mu_beta_0",),
            ("z_mu_beta_1",),
            *block_diag_slope,
            *block_diag_int,
        ]

    def numpyro_model(self):
        sigma_beta_0 = 1.0
        sigma_beta_1 = 0.25

        sigma_global_beta_0 = jnp.sqrt(self.pool_beta_0 * sigma_beta_0)
        sigma_global_beta_1 = jnp.sqrt(self.pool_beta_1 * sigma_beta_1)

        sigma_local_beta_0 = jnp.sqrt((1.0 - self.pool_beta_0) * sigma_beta_0)
        sigma_local_beta_1 = jnp.sqrt(
            (1.0 - self.pool_beta_1) * sigma_beta_1
        ) * np.ones(self.num_lineages)

        z_mu_beta_0 = numpyro.sample("z_mu_beta_0", dist.Normal(0, 1))

        z_mu_beta_1 = numpyro.sample("z_mu_beta_1", dist.Normal(0, 1))

        mu_beta_0 = numpyro.deterministic(
            "mu_beta_0", z_mu_beta_0 * sigma_global_beta_0
        )
        mu_beta_1 = numpyro.deterministic(
            "mu_beta_1", z_mu_beta_1 * sigma_global_beta_1
        )

        omega_decomposition = numpyro.sample(
            "omega_decomposition",
            dist.LKJCholesky(self.num_lineages, 2),
        )
        # A faster version of `np.diag(sigma_beta_1) @ Omega_decomposition`
        sigma_decomposition_t = (
            sigma_local_beta_1[:, None] * omega_decomposition
        ).T

        v_z_0 = []
        v_z_1 = []

        for i in range(np.unique(self.divisions).size):
            z_0 = numpyro.sample(
                f"z_0_{i}",
                dist.Normal(0.0, 1.0),
                sample_shape=(self.num_lineages,),
            )

            z_1 = numpyro.sample(
                f"z_1_{i}",
                dist.Normal(0.0, 1.0),
                sample_shape=(self.num_lineages,),
            )

            v_z_0.append(z_0)
            v_z_1.append(z_1)

        beta_0 = numpyro.deterministic(
            "beta_0",
            mu_beta_0 + (sigma_local_beta_0 * jnp.column_stack(v_z_0)).T,
        )

        beta_1 = numpyro.deterministic(
            "beta_1",
            mu_beta_1 + (jnp.column_stack(v_z_1).T @ sigma_decomposition_t),
        )

        likelihood = multinomial_likelihood(
            beta_0, beta_1, self.divisions, self.time, self.N
        )

        # Y[i, l] is the count of lineage l for observation i
        numpyro.sample("Y", likelihood, obs=self.counts)


class IndependentDivisionsModel(MultinomialModel):
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

        self.time = data["fd_offset"].to_numpy()

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

    def dense_mass(self):
        block_diag_int = [
            (f"v_z_0_{grp}",) for grp in range(np.unique(self.divisions).size)
        ]
        block_diag_slope = [
            (f"v_z_1_{grp}",) for grp in range(np.unique(self.divisions).size)
        ]
        return [*block_diag_slope, *block_diag_int]

    def numpyro_model(self):
        v_beta_0 = []
        v_beta_1 = []
        sigma_0 = 1.0
        sigma_1 = 0.25

        for i in range(np.unique(self.divisions).size):
            v_z_0 = numpyro.sample(
                f"v_z_0_{i}",
                dist.Normal(0.0, 1.0),
                sample_shape=(self.num_lineages,),
            )

            v_z_1 = numpyro.sample(
                f"v_z_1_{i}",
                dist.Normal(0.0, 1.0),
                sample_shape=(self.num_lineages,),
            )

            v_beta_0.append(v_z_0 * sigma_0)
            v_beta_1.append(v_z_1 * sigma_1)

        beta_0 = numpyro.deterministic("beta_0", jnp.column_stack(v_beta_0).T)

        beta_1 = numpyro.deterministic("beta_1", jnp.column_stack(v_beta_1).T)

        likelihood = multinomial_likelihood(
            beta_0, beta_1, self.divisions, self.time, self.N
        )

        # Y[i, l] is the count of lineage l for observation i
        numpyro.sample("Y", likelihood, obs=self.counts)

    def create_forecasts(self, mcmc, fd_offsets) -> pl.DataFrame:
        parameter_samples = (
            expand_grid(
                chain=np.arange(mcmc.num_chains),
                iteration=np.arange(mcmc.num_samples // mcmc.thinning),
                division=self.division_names,
                lineage=self.lineage_names,
            )
            .with_columns(
                beta_0=np.asarray(mcmc.get_samples()["beta_0"]).flatten(),
                beta_1=np.asarray(mcmc.get_samples()["beta_1"]).flatten(),
                sample_index=pl.col("iteration")
                + pl.col("chain") * (mcmc.num_samples // mcmc.thinning)
                + 1,
            )
            .drop("chain", "iteration")
        )

        return ForecastFrame(
            expand_grid(
                sample_index=parameter_samples["sample_index"].unique(),
                fd_offset=fd_offsets,
            )
            .join(parameter_samples, on="sample_index")
            .with_columns(
                phi=pl_softmax(
                    pl.col("beta_0") + pl.col("beta_1") * pl.col("fd_offset"),
                ).over("sample_index", "division", "fd_offset")
            )
            .drop("beta_0", "beta_1")
        )


class BaselineModel(MultinomialModel):
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
                iteration=np.arange(mcmc.num_samples // mcmc.thinning),
                division=self.division_names,
                lineage=self.lineage_names,
            )
            .with_columns(
                logit_phi=np.asarray(
                    mcmc.get_samples()["logit_phi"]
                ).flatten(),
                sample_index=pl.col("iteration")
                + pl.col("chain") * (mcmc.num_samples // mcmc.thinning)
                + 1,
            )
            .drop("chain", "iteration")
        )

        return ForecastFrame(
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
