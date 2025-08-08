import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from linmod.data import CountsFrame
from linmod.models import ForecastFrame
from linmod.utils import pl_list_cycle, pl_norm


def optional_filter(df: pl.LazyFrame | pl.DataFrame, filters: dict | None):
    """
    Filters the data based on a dict of column name to allowable values.
    `None` is interpreted as all values allowed
    """
    if filters is None:
        return df
    else:
        for col, vals in filters.items():
            assert col in df.lazy().collect_schema().names()
            if vals is not None:
                df = df.filter(pl.col(col).is_in(vals))

    return df


def multinomial_count_sampler(
    n: ArrayLike,
    p: ArrayLike,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Samples from a `multinomial(n, p)` distribution.

    Compatible shapes of `n` and `p` include:
    - `n` is a scalar, `p` is a vector
    - `n` is a vector, `p` is a matrix with rows corresponding to entries in `n`
    """

    return rng.multinomial(n, p)


class ProportionsEvaluator:
    def __init__(self, samples: ForecastFrame, data: CountsFrame):
        assert (
            samples["lineage"].unique().sort()
            == data["lineage"].unique().sort()
        ).all()

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
            .filter(pl.col("phi").is_not_nan())
            .join(
                samples,
                on=("fd_offset", "division", "lineage"),
                how="left",
                suffix="_sampled",
            )
        )

        assert (
            self.df["fd_offset"].unique().sort()
            == data["fd_offset"].unique().sort()
        ).all()

        self.df = self.df.lazy()

    def _mean_norm_per_division_day(self, p=1):
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

    def mean_norm(self, filters=None, p=1) -> float:
        r"""
        The expected norm of proportion forecast error, summed over all divisions
        and days.

        $\sum_{t, g} E[ || f_{tg} - \phi_{tg} ||_p ]$
        """

        return (
            self._mean_norm_per_division_day(p=p)
            .pipe(optional_filter, filters=filters)
            .collect()
            .get_column("mean_norm")
            .sum()
        )


class CountsEvaluator:
    _count_samplers = {
        "multinomial": multinomial_count_sampler,
    }

    def __init__(
        self,
        samples: ForecastFrame,
        data: CountsFrame,
        count_sampler: str = "multinomial",
        seed: int | None = None,
    ) -> None:
        r"""
        Evaluates count forecasts $\hat{Y}$ sampled from a specified observation model given model
        proportion forecasts.

        `count_sampler` should be one of the keys in `CountsEvaluator._count_samplers`.
        `seed` is an optional random seed for the count sampler.
        """

        assert count_sampler in type(self)._count_samplers, (
            f"Count sampler '{count_sampler}' not found. "
            f"Available samplers: {', '.join(type(self)._count_samplers)}"
        )
        count_sampler = type(self)._count_samplers[count_sampler]

        assert (
            samples["lineage"].unique().sort()
            == data["lineage"].unique().sort()
        ).all()

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

        assert (
            self.df["fd_offset"].unique().sort()
            == data["fd_offset"].unique().sort()
        ).all()

        self.df = self.df.lazy()

    def _uncovered_per_lineage_division_day(self, alpha=0.05):
        """
        For each lineage in each division on each day, False if the observed count in the
        (1 - alpha) x 100% univariate prediction interval, True otherwise.
        """
        q_low = alpha / 2
        q_high = 1.0 - q_low

        return (
            self.df.group_by("fd_offset", "division", "lineage")
            .agg(
                lci=pl.col("count_sampled").quantile(q_low),
                uci=pl.col("count_sampled").quantile(q_high),
                count=pl.col("count").min(),  # all counts are the same
            )
            .with_columns(
                uncovered=(
                    (pl.col("count") < pl.col("lci"))
                    | (pl.col("count") > pl.col("uci"))
                )
            )
        )

    def uncovered_proportion(self, filters=None, alpha=0.05) -> float:
        """
        Proportion of all lineage observation counts on all division-days not covered
        by the (central) 1 - alpha prediction interval.
        """
        prop = (
            self._uncovered_per_lineage_division_day(alpha)
            .pipe(optional_filter, filters=filters)
            .collect()
            .get_column("uncovered")
            .cast(pl.Int8)
            .mean()
        )
        assert isinstance(prop, float)
        return prop

    def _mean_norm_per_division_day(self, p=1):
        r"""
        The expected norm of count forecast error for each division-day.

        $E[ || \hat{Y}_{tg} - Y_{tg} ||_p ]$

        Returns a data frame with columns `(division, fd_offset, mean_norm)`.
        """

        return (
            self.df.group_by("fd_offset", "division", "sample_index")
            .agg(norm=pl_norm(pl.col("count") - pl.col("count_sampled"), p))
            .group_by("fd_offset", "division")
            .agg(mean_norm=pl.mean("norm"))
        )

    def mean_norm(self, filters=None, p=1) -> float:
        r"""
        The expected norm of count forecast error, summed over all divisions
        and days.

        $\sum_{t, g} E[ || \hat{Y}_{tg} - Y_{tg} ||_p ]$
        """

        return (
            self._mean_norm_per_division_day(p=p)
            .pipe(optional_filter, filters=filters)
            .collect()
            .get_column("mean_norm")
            .sum()
        )

    def _energy_score_per_division_day(self, p=2):
        r"""
        Monte Carlo approximation to the energy score (multivariate generalization of
        CRPS) of count forecasts for each division-day.

        $E[ || \hat{Y}_{tg} - Y_{tg} ||_p ] - \frac{1}{2} E[ || \hat{Y}_{tg} - \hat{Y}_{tg}' ||_p ]$

        Returns a data frame with columns `(division, fd_offset, energy_score)`.
        """

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

    def energy_score(self, filters=None, p=2) -> float:
        r"""
        The energy score of count forecasts, summed over all divisions and days.

        $$
        \sum_{t, g} E[ || \hat{Y}_{tg} - Y_{tg} ||_p ]
        - \frac{1}{2} E[ || \hat{Y}_{tg} - \hat{Y}_{tg}' ||_p ]
        $$
        """
        return (
            self._energy_score_per_division_day(p=p)
            .pipe(optional_filter, filters=filters)
            .collect()
            .get_column("energy_score")
            .sum()
        )
