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


def plot_forecast(forecast, counts=None):
    summaries = forecast.group_by("division", "fd_offset", "lineage").agg(
        mean_phi=pl.mean("phi"),
        q_lower=pl.quantile("phi", 0.1),
        q_upper=pl.quantile("phi", 0.9),
    )

    if counts is not None:
        day_counts = (
            counts.group_by(pl.col("fd_offset"), pl.col("division"))
            .agg(pl.col("count").sum().alias("n_obs"))
            .select("fd_offset", "division", "n_obs")
        )
        counts = counts.with_columns(
            f=(
                pl.col("count") / pl.sum("count").over("fd_offset", "division")
            ),
        ).join(
            day_counts,
            on=("fd_offset", "division"),
            how="left",
            validate="m:1",
        )
        summaries = summaries.join(
            counts,
            on=("fd_offset", "division", "lineage"),
            how="left",
            validate="1:1",
        )

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
        + theme_bw(base_size=20)
        + ylab("phi")
    )
    if counts is not None:
        plot = (
            plot
            + geom_point(
                aes(
                    "fd_offset",
                    y="f",
                    color="lineage",
                    fill="lineage",
                    size="count",
                ),
                alpha=0.5,
                shape="o",
            )
            + scale_size(
                range=(0.1, 3),
            )
        )

    return plot
