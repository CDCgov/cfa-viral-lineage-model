from functools import reduce

import polars as pl


# Like the R version
def expand_grid(**columns):
    column_dfs = map(
        lambda c: pl.DataFrame(c[1], schema=[c[0]]), columns.items()
    )
    df = reduce(lambda x, y: x.join(y, how="cross"), column_dfs)

    return df.sort(columns.keys())


def pl_softmax(pl_expr, over=None):
    if over is None:
        return pl_expr.exp() / pl_expr.exp().sum()
    else:
        return pl_expr.exp() / pl_expr.exp().sum().over(over)
