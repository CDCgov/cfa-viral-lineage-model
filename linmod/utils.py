import sys
from functools import reduce
from pathlib import Path

import numpy as np
import polars as pl


class ValidPath(Path):
    """
    Functions like `pathlib.Path`, but ensures that the path exists.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.is_dir():
            self.mkdir(parents=True, exist_ok=True)
        else:
            self.parent.mkdir(parents=True, exist_ok=True)

    def __truediv__(self, child):
        return ValidPath(super().__truediv__(child))


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


def read_phi(fp: str | Path):
    """
    Reads forecasted proportions and formats into an ndarray of (posterior sample index, group x time, lineage)
    """
    df = pl.read_csv(fp)
    assert set(
        ["sample_index", "fd_offset", "division", "lineage", "phi"]
    ) == set(df.columns)

    indices = df["sample_index"].unique().sort()

    return np.array(
        [
            (
                df.filter(pl.col("sample_index") == i)
                .pivot(
                    on="lineage", values="phi", index=["division", "fd_offset"]
                )
                .drop("division", "fd_offset")
            ).to_numpy()
            for i in indices
        ]
    )
