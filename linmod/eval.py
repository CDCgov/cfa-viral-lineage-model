import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from linmod.data import CountsFrame
from linmod.models import ForecastFrame
from linmod.utils import pl_norm


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
    Samples from multinomial for multiple rows in one call.

    n: 1-D array-like of total counts for each draw.
    p: 2-D array-like where each row is a probability vector for that draw.
    rng: a numpy.random.Generator used for reproducible sampling.

    Returns an (n_rows, n_lineages) ndarray of integer counts.
    """

    assert isinstance(n, np.ndarray)
    assert isinstance(p, np.ndarray)

    assert n.ndim == 1
    assert p.ndim == 2

    if p.shape[0] != n.shape[0]:
        raise ValueError(
            "Length of n must match number of probability vectors in p"
        )

    draws = rng.multinomial(n, p)

    return draws


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

        cols = samples.collect_schema().names()
        groups = ["date", "fd_offset", "division", "sample_index"] + (
            ["chain", "iteration"]
            if (("chain" in cols) and ("iteration" in cols))
            else []
        )

        grouped = (
            data.join(
                samples.rename({"phi": "phi_sampled"}),
                on=("fd_offset", "division", "lineage"),
                how="left",
            )
            .group_by(groups)
            .agg(pl.col("lineage"), pl.col("phi_sampled"), pl.col("count"))
        )

        # Sort deterministically for reproducible RNG draws
        grouped = grouped.sort(by=groups)

        rng = np.random.default_rng(seed)

        count_tots = np.sum(np.array(grouped["count"].to_list()), axis=1)

        sampled_array = count_sampler(
            count_tots, np.array(grouped["phi_sampled"].to_list()), rng
        )

        grouped = grouped.with_columns(count_sampled=pl.Series(sampled_array))

        # Now explode to long format analogous to previous pipeline
        df = grouped.explode(
            "lineage", "phi_sampled", "count", "count_sampled"
        ).drop("phi_sampled")

        # Keep the rest of pipeline consistent with previous behaviour
        self.df = df.lazy()

    def _uncovered_per_lineage_division_day(self, alpha=0.11):
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

    def uncovered_proportion(self, filters=None, alpha=0.11) -> float:
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

        This function depends upon having multiple _independent_ MCMC chains, which
        it uses as the "replicate" data Y' when computing the second expectation.
        """
        cols = self.df.collect_schema().names()

        assert (
            "chain" in cols
        ), "Cannot compute Energy Score without chain IDs."
        assert (
            "iteration" in cols
        ), "Cannot compute Energy Score without within-chain iteration IDs."

        chain_index = self.df.select("chain").collect()["chain"]
        chains = sorted(chain_index.unique().to_list())
        nchains_2 = len(chains) // 2
        chains_left = chains[:nchains_2]
        chains_right = chains[nchains_2 : (nchains_2 * 2)]
        chain_rename = {
            right: left for left, right in zip(chains_left, chains_right)
        }

        df = self.df.filter(pl.col("chain").is_in(chains_left)).join(
            self.df.filter(pl.col("chain").is_in(chains_right))
            .with_columns(
                pl.col("count_sampled").alias("replicate"),
                pl.col("chain").replace_strict(chain_rename),
            )
            .drop(["sample_index", "count", "count_sampled"]),
            on=[
                "date",
                "fd_offset",
                "division",
                "lineage",
                "chain",
                "iteration",
            ],
            how="inner",
            validate="1:1",
        )

        return (
            # First, we will gather the values of count' we will use for (count-count')
            df.group_by("date", "fd_offset", "division", "lineage")
            .agg(
                pl.col("sample_index"),
                pl.col("count"),
                pl.col("count_sampled"),
                pl.col("replicate"),
            )
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
