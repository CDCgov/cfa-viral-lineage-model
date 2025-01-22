import string
from abc import ABC, abstractmethod
from itertools import product
from typing import Iterable

import polars as pl
from numpyro.diagnostics import summary
from numpyro.infer.mcmc import MCMC
from plotnine import aes, geom_line, ggplot, theme_bw


class ForecastFrame(pl.DataFrame):
    """
    A `polars.DataFrame` which enforces a format for probabilistic forecast samples of
    population-level lineage proportions.

    See `REQUIRED_COLUMNS` for the expected columns.
    """

    REQUIRED_COLUMNS = {
        "sample_index",
        "fd_offset",
        "division",
        "lineage",
        "phi",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.validate()

    @classmethod
    def read_parquet(cls, *args, **kwargs):
        return cls(pl.read_parquet(*args, **kwargs))

    def validate(self, *args, **kwargs):
        # In case polars ever adds a validate method
        if hasattr(super(), "validate"):
            super().validate(*args, **kwargs)

        assert self.REQUIRED_COLUMNS.issubset(
            self.columns
        ), f"Missing at least one required column ({', '.join(self.REQUIRED_COLUMNS)})"

        proportion_sums = self.group_by(
            "sample_index", "fd_offset", "division"
        ).agg(pl.sum("phi"))

        assert (
            (proportion_sums["phi"] - 1).abs() < 1e-3
        ).all(), f"Lineage proportions do not sum to 1."


class GeographicAggregator(ABC):
    r"""
    Aggregate forecasts from state level to some larger grouping, like HHS divisions.
    """

    @abstractmethod
    def __call__(
        self, forecast: pl.DataFrame, geo_map: dict[str, str], **kwargs
    ) -> pl.DataFrame:
        raise NotImplementedError()


class InfectionWeightedAggregator(GeographicAggregator):
    r"""
    Geographically aggregate while accounting for unequal numbers of infected individuals in different states.

    For convenience, this is broken down into per-state total human populations, and the proportions of those populations
    which are infected. The per-state weight in the aggregation is the product of these terms.

    The state-level populations and proportions infected are taken to be constant over the forecasting horizon.

    __call__ arguments
    forecast:          A LazyFrame with the standard model output format.
    geo_map:           A dictionary mapping "divisions" in the forecast to new units.
                       Every existing division need not be mapped.

    __call__ keyword arguments
    pop_size:          A pl.LazyFrame with the (possibly relative) population size of each forecasted division.
                       Must contain all divisions which are in the `geo_map`, and two columns: "division" and "pop_size".
                       Defaults to equal population sizes.
    prop_infected      A pl.LazyFrame with the proportion of the populations infected (or something proportionate thereto),
                       taken to be a constant over the modeling period. Must contain all divisions which are in the `geo_map`,
                       and two columns: "division" and "prop_infected". Defaults to equal proportions infected.

    Returns a LazyFrame in the standard output format on the new geographical units.
    """

    def __call__(
        self, forecast: pl.DataFrame, geo_map: dict[str, str], **kwargs
    ) -> pl.DataFrame:
        pop_size = kwargs.get(
            "pop_size",
            pl.DataFrame(
                {
                    "division": geo_map.keys(),
                    "pop_size": [1.0] * len(geo_map.keys()),
                }
            ),
        )
        prop_infected = kwargs.get(
            "prop_infected",
            pl.DataFrame(
                {
                    "division": geo_map.keys(),
                    "prop_infected": [1.0] * len(geo_map.keys()),
                }
            ),
        )

        assert set(geo_map.keys()).issubset(
            set(forecast["division"].unique())
        ), 'All divisions in `geo_map.keys()` must be in `forecast["division"].'

        assert set(geo_map.keys()).issubset(
            set(pop_size["division"])
        ), 'All divisions in `geo_map.keys()` must be in `pop_size["division"].'

        assert set(geo_map.keys()).issubset(
            set(prop_infected["division"])
        ), 'All divisions in `geo_map.keys()` must be in `prop_infected["division"].'

        weights = (
            pop_size.join(
                prop_infected, on="division", how="inner", validate="1:1"
            )
            .with_columns(
                weight=(pl.col("pop_size") * pl.col("prop_infected"))
            )
            .filter(pl.col("division").is_in(geo_map.keys()))
            .select("division", "weight")
        )

        return (
            forecast.filter(pl.col("division").is_in(geo_map.keys()))
            .with_columns(
                pl.col("division")
                .replace_strict(geo_map)
                .alias("new_division")
            )
            .join(weights, on="division", how="left", validate="m:1")
            .with_columns(pl.col("phi") * pl.col("weight"))
            .group_by("sample_index", "fd_offset", "new_division", "lineage")
            .agg(pl.sum("phi"))
            .with_columns(
                pl.col("phi")
                / pl.sum("phi").over(
                    "sample_index", "fd_offset", "new_division"
                )
            )
            .rename({"new_division": "division"})
            .sort("sample_index", "division", "lineage", "fd_offset")
        )


def get_convergence(
    mcmc: MCMC, ignore_nan_in: list[str] = [], drop_ignorable_nan: bool = True
) -> pl.DataFrame:
    allowed = set(string.ascii_lowercase + string.ascii_uppercase + "_")
    for ignore in ignore_nan_in:
        assert set(ignore).issubset(allowed)

    raw_summary = summary(mcmc.get_samples(group_by_chain=True))

    convergence = pl.DataFrame(
        {
            "param": [],
            "param_no_dim": [],
            "n_eff": [],
            "r_hat": [],
        }
    ).cast(
        {
            "param": pl.String,
            "param_no_dim": pl.String,
            "n_eff": pl.Float64,
            "r_hat": pl.Float64,
        }
    )

    for k, v in raw_summary.items():
        names_nodim = []
        names = []
        ess = []
        psrf = []

        if len(v["n_eff"].shape) == 0:
            names = [k]
            names_nodim = [k]
            ess = [float(v["n_eff"])]
            psrf = [float(v["r_hat"])]
        else:
            array_indices = [
                idx for idx in product(*map(range, v["n_eff"].shape))
            ]
            names = [
                k + "[" + ",".join([str(i) for i in idx]) + "]"
                for idx in array_indices
            ]
            names_nodim = [k] * len(array_indices)
            ess = [float(v["n_eff"][idx]) for idx in array_indices]
            psrf = [float(v["r_hat"][idx]) for idx in array_indices]

        param = pl.DataFrame(
            {
                "param": names,
                "param_no_dim": names_nodim,
                "n_eff": ess,
                "r_hat": psrf,
            }
        )

        convergence = pl.concat([convergence, param])

    nans = convergence.filter(
        (pl.col("n_eff").is_nan()) | (pl.col("r_hat").is_nan())
    )
    if nans.shape[0] > 0:
        nans = nans.filter(~pl.col("param_no_dim").is_in(ignore_nan_in))
        if nans.shape[0] > 0:
            bad = nans["param"].unique().to_list()
            raise RuntimeError(
                "Found unexpected NaN convergence values for parameters: "
                + str(bad)
            )
        if drop_ignorable_nan:
            convergence = convergence.filter(
                (pl.col("n_eff").is_not_nan()) & (pl.col("r_hat").is_not_nan())
            )

    return convergence.drop("param_no_dim")


def plot_convergence(mcmc: MCMC, params: Iterable[str]):
    posterior = mcmc.get_samples(group_by_chain=True)
    plots = []
    for par in params:
        if par.count("[") > 0:
            dimless_par = par.split("[")[0]
            indices = [int(i) for i in par.split("[")[1][:-1].split(",")]
            index_tuple = tuple(
                [*[slice(None), slice(None)], *[i for i in indices]]
            )
            chains = posterior[dimless_par][index_tuple]
        else:
            chains = posterior[par]

        niter = chains.shape[1]
        df = pl.concat(
            [
                pl.DataFrame(
                    {
                        "param": [par] * niter,
                        "chain": [str(chain)] * niter,
                        "iteration": list(range(niter)),
                        "value": chains[chain].tolist(),
                    }
                )
                for chain in range(chains.shape[0])
            ]
        )

        plt = (
            ggplot(df)
            + geom_line(
                aes(
                    x="iteration",
                    y="value",
                    color="chain",
                ),
            )
            + theme_bw(base_size=20)
        )

        plots.append(plt)

    return plots
