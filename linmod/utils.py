from functools import reduce

import polars as pl


# Like the R version
def expand_grid(**columns):
    column_dfs = map(
        lambda c: pl.DataFrame(c[1], schema=[c[0]]), columns.items()
    )
    df = reduce(lambda x, y: x.join(y, how="cross"), column_dfs)

    return df.sort(columns.keys())


def pl_crps(samples_column: str, truth_column: str):
    """
    Monte Carlo approximation to the CRPS.
    """

    samples = pl.col(samples_column)
    truth = pl.col(truth_column)
    n = samples.len()

    return (samples - truth).abs().mean() - 0.5 * (
        samples.head(n - 1) - samples.tail(n - 1)
    ).abs().mean()


def pl_softmax(pl_expr):
    return pl_expr.exp() / pl_expr.exp().sum()


def pl_mae(samples_column: str, truth_column: str):
    return (pl.col(samples_column) - pl.col(truth_column)).abs().mean()
