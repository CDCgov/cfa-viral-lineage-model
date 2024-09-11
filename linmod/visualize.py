import polars as pl
from plotnine import (
    aes,
    facet_wrap,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    scale_size,
    theme_bw,
    ylab,
)


def plot_forecast(
    forecast: pl.LazyFrame | pl.DataFrame,
    counts: pl.DataFrame = None,
    base_size=20,
):
    summaries = forecast.group_by("division", "fd_offset", "lineage").agg(
        mean_phi=pl.mean("phi"),
        q_lower=pl.quantile("phi", 0.1),
        q_upper=pl.quantile("phi", 0.9),
    )

    if isinstance(forecast, pl.LazyFrame):
        summaries = summaries.collect()

    plot = (
        ggplot(summaries)
        + geom_ribbon(
            aes(
                "fd_offset",
                ymin="q_lower",
                ymax="q_upper",
                group="lineage",
                fill="lineage",
            ),
            alpha=0.15,
        )
        + geom_line(
            aes("fd_offset", "mean_phi", group="lineage", color="lineage"),
            size=1.5,
        )
        + facet_wrap("division")
        + theme_bw(base_size=base_size)
        + ylab("phi")
    )

    if counts is not None:
        counts = counts.with_columns(
            f=pl.col("count") / pl.sum("count").over("fd_offset", "division"),
            n_obs=pl.sum("count").over("fd_offset", "division"),
        )

        plot = (
            plot
            + geom_point(
                aes(
                    "fd_offset",
                    y="f",
                    color="lineage",
                    fill="lineage",
                    size="n_obs",
                ),
                data=counts,
                alpha=0.5,
                shape="o",
            )
            + scale_size(
                range=(0.1, 3),
            )
        )

    return plot
