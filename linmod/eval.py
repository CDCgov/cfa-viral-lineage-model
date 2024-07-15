import polars as pl
from plotnine import (
    aes,
    facet_wrap,
    geom_line,
    geom_ribbon,
    ggplot,
    theme_bw,
    ylab,
)


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


def dominating_lineages(df: pl.DataFrame, threshold: float):
    """
    In each division, return (samples from) the joint distribution of
    the lineage that first reaches a certain threshold of prevalance
    and the day it does so.
    """

    return (
        df.filter(pl.col("phi") > threshold, pl.col("day") > 0)
        .group_by("sample_index", "division", "lineage")
        .agg(pl.min("day"))
        .sort("division")
        .drop("sample_index")
    )


def plot_samples(samples):
    summaries = samples.group_by("division", "day", "lineage").agg(
        mean_phi=pl.mean("phi"),
        q_lower=pl.quantile("phi", 0.1),
        q_upper=pl.quantile("phi", 0.9),
    )

    return (
        ggplot(summaries)
        + geom_ribbon(
            aes(
                "day",
                ymin="q_lower",
                ymax="q_upper",
                group="lineage",
                fill="lineage",
            ),
            alpha=0.15,
        )
        + geom_line(
            aes("day", "mean_phi", group="lineage", color="lineage"),
            size=1.5,
        )
        + facet_wrap("division")
        + theme_bw(base_size=20)
        + ylab("phi")
    )
