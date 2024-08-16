import string
import sys
from functools import reduce
from itertools import product
from typing import Iterable

import polars as pl
from numpyro.diagnostics import summary
from numpyro.infer.mcmc import MCMC
from plotnine import aes, geom_line, ggplot, theme_bw


def expand_grid(**columns):
    """
    Create a DataFrame from all combinations of given columns.

    Operates like the R function `tidyr::expand_grid`.
    """

    column_dfs = map(
        lambda c: pl.DataFrame(c[1], schema=[c[0]]), columns.items()
    )
    df = reduce(lambda x, y: x.join(y, how="cross"), column_dfs)

    return df.sort(columns.keys())


def print_message(*args, **kwargs):
    """
    Print a message to stderr and flush the buffer.
    """

    print(*args, **kwargs, file=sys.stderr)
    sys.stderr.flush()


def pl_list_cycle(pl_expr, n: int):
    """
    Returns the column computed by `pl_expr`, but with the last `n` elements
    moved to the front.
    """

    assert n > 0

    return pl_expr.list.tail(n).list.concat(
        pl_expr.list.head(pl_expr.list.len() - n)
    )


def pl_norm(pl_expr, p: int):
    r"""
    Computes the L_p norm $||\cdot||_p$ of the column `pl_expr`.
    """
    return pl_expr.abs().pow(p).sum().pow(1 / p)


def pl_softmax(pl_expr):
    """
    Computes the softmax of the column `pl_expr`.
    """
    return pl_expr.exp() / pl_expr.exp().sum()


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
