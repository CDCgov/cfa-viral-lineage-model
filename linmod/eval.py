import numpy as np
import polars as pl

from linmod.utils import pl_list_cycle, pl_norm


def multinomial_count_sampler(N, phi, rng):
    return rng.multinomial(N, phi)


class ProportionsEvaluator:
    def __init__(self, samples: pl.LazyFrame, data: pl.LazyFrame):
        """
        `samples` should have the standard model output format.
        `data` should have the standard model input format.
        """

        # Join the forecast samples and raw data dataframes.
        # Also compute the true proportions from the raw data.
        self.df = (
            data.with_columns(
                phi=(
                    pl.col("count")
                    / pl.sum("count").over("fd_offset", "division")
                ),
            )
            .drop("count")
            # NaN will appear on division-days where none of the lineages
            # were sampled. We will filter these out.
            # TODO: Is this good to do?
            .filter(pl.col("phi").is_not_nan())
            .join(
                samples,
                on=("fd_offset", "division", "lineage"),
                how="left",
                suffix="_sampled",
            )
        )

    def _mean_norm_per_division_day(self, p=1) -> pl.LazyFrame:
        r"""
        The expected norm of proportion forecast error for each division-day.

        $E[ || f_{tg} - \phi_{tg} ||_p ]$

        Returns a data frame with columns `(division, fd_offset, mean_norm)`.
        """

        return (
            self.df.group_by("fd_offset", "division", "sample_index")
            .agg(norm=pl_norm(pl.col("phi") - pl.col("phi_sampled"), p))
            .group_by("fd_offset", "division")
            .agg(mean_norm=pl.mean("norm"))
        )

    def mean_norm(self, p=1) -> float:
        r"""
        The expected norm of proportion forecast error, summed over all divisions
        and days.

        $\sum_{t, g} E[ || f_{tg} - \phi_{tg} ||_p ]$
        """

        return (
            self._mean_norm_per_division_day(p=p)
            .collect()
            .get_column("mean_norm")
            .sum()
        )

    def _energy_score_per_division_day(self, p=2) -> pl.LazyFrame:
        r"""
        Monte Carlo approximation to the energy score (multivariate generalization of
        CRPS) of proportion forecasts for each division-day.

        $E[ || f_{tg} - \phi_{tg} ||_p ] - \frac{1}{2} E[ || f_{tg} - \f_{tg}' ||_p ]$

        Returns a data frame with columns `(division, fd_offset, energy_score)`.
        """

        return (
            # First, we will gather the values of phi' we will use for (phi-phi')
            self.df.group_by("date", "fd_offset", "division", "lineage")
            .agg(pl.col("sample_index"), pl.col("phi"), pl.col("phi_sampled"))
            .with_columns(replicate=pl_list_cycle(pl.col("phi_sampled"), 1))
            .explode("sample_index", "phi", "phi_sampled", "replicate")
            # Now we can compute the score
            .group_by("fd_offset", "division", "sample_index")
            .agg(
                term1=pl_norm(pl.col("phi") - pl.col("phi_sampled"), p),
                term2=pl_norm(
                    (pl.col("phi_sampled") - pl.col("replicate")), p
                ),
            )
            .group_by("fd_offset", "division")
            .agg(
                energy_score=pl.col("term1").mean()
                - 0.5 * pl.col("term2").mean()
            )
        )

    def energy_score(self, p=2) -> float:
        r"""
        The energy score of proportion forecasts, summed over all divisions and days.

        $$
        \sum_{t, g} E[ || f_{tg} - \phi_{tg} ||_p ]
        - \frac{1}{2} E[ || f_{tg} - f_{tg}' ||_p ]
        $$
        """

        return (
            self._energy_score_per_division_day(p=p)
            .collect()
            .get_column("energy_score")
            .sum()
        )


class CountsEvaluator:
    _count_samplers = {
        "multinomial": multinomial_count_sampler,
    }

    def __init__(
        self,
        samples: pl.LazyFrame,
        data: pl.LazyFrame,
        count_sampler: str = "multinomial",
        seed: int = 42,
    ):
        assert count_sampler in type(self)._count_samplers, (
            f"Count sampler '{count_sampler}' not found. "
            f"Available samplers: {', '.join(type(self)._count_samplers)}"
        )
        count_sampler = type(self)._count_samplers[count_sampler]

        rng = np.random.default_rng(seed)

        self.df = (
            data.join(
                samples.rename({"phi": "phi_sampled"}),
                on=("fd_offset", "division", "lineage"),
                how="left",
            )
            .group_by("date", "fd_offset", "division", "sample_index")
            .agg(pl.col("lineage"), pl.col("phi_sampled"), pl.col("count"))
            .with_columns(
                count_sampled=pl.struct(
                    pl.col("phi_sampled"), N=pl.col("count").list.sum()
                ).map_elements(
                    lambda struct: list(
                        count_sampler(struct["N"], struct["phi_sampled"], rng)
                    ),
                    return_dtype=pl.List(pl.Int64),
                )
            )
            .explode("lineage", "phi_sampled", "count", "count_sampled")
            .drop("phi_sampled")
        )

    def _mean_norm_per_division_day(self, p=1) -> pl.LazyFrame:
        return (
            self.df.group_by("fd_offset", "division", "sample_index")
            .agg(norm=pl_norm(pl.col("count") - pl.col("count_sampled"), p))
            .group_by("fd_offset", "division")
            .agg(mean_norm=pl.mean("norm"))
        )

    def mean_norm(self, p=1) -> float:
        return (
            self._mean_norm_per_division_day(p=p)
            .collect()
            .get_column("mean_norm")
            .sum()
        )

    def _energy_score_per_division_day(self, p=2) -> pl.LazyFrame:
        return (
            # First, we will gather the values of count' we will use for (count-count')
            self.df.group_by("date", "fd_offset", "division", "lineage")
            .agg(
                pl.col("sample_index"),
                pl.col("count"),
                pl.col("count_sampled"),
            )
            .with_columns(replicate=pl_list_cycle(pl.col("count_sampled"), 1))
            .explode("sample_index", "count", "count_sampled", "replicate")
            # Now we can compute the score
            .group_by("fd_offset", "division", "sample_index")
            .agg(
                term1=pl_norm(pl.col("count") - pl.col("count_sampled"), p),
                term2=pl_norm(
                    (pl.col("count_sampled") - pl.col("replicate")), p
                ),
            )
            .group_by("fd_offset", "division")
            .agg(
                energy_score=pl.col("term1").mean()
                - 0.5 * pl.col("term2").mean()
            )
        )

    def energy_score(self, p=2) -> float:
        return (
            self._energy_score_per_division_day(p=p)
            .collect()
            .get_column("energy_score")
            .sum()
        )
