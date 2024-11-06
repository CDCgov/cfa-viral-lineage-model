import argparse
from typing import Optional

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

from .utils import ValidPath


def plot_forecast(
    forecast: pl.LazyFrame | pl.DataFrame,
    counts: Optional[pl.DataFrame] = None,
    base_size=20,
):
    summaries = forecast.group_by("division", "fd_offset", "lineage").agg(
        mean_phi=pl.mean("phi"),
        q_lower=pl.quantile("phi", 0.1),
        q_upper=pl.quantile("phi", 0.9),
    )

    if isinstance(summaries, pl.LazyFrame):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python3 -m linmod.visualize",
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--forecast",
        type=str,
        help="Path to forecast parquet file",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Path to forecasts parquet file",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="eval|model|nodata, to plot evaluation or modeling or no data",
    )
    parser.add_argument(
        "-p",
        "--png",
        type=str,
        help="Path to desired path to save PNG",
    )
    args = parser.parse_args()

    forecast = pl.read_parquet(args.forecast)
    lineages = forecast["lineage"].unique()

    if args.type == "eval":
        fd_filter = pl.col("fd_offset") > 0
    elif args.type == "model":
        fd_filter = pl.col("fd_offset") <= 0
    elif not args.type == "nodata":
        raise RuntimeError("Invalid type of plot.")

    data = None
    if (args.type != "nodata") and (args.data is not None):
        data = pl.read_parquet(args.data).filter(
            pl.col("lineage").is_in(lineages), fd_filter
        )

    plot_forecast(forecast, data).save(
        ValidPath(args.png),
        width=25,
        height=15,
        dpi=200,
        verbose=False,
    )
