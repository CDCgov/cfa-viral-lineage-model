from functools import reduce

import polars as pl


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
