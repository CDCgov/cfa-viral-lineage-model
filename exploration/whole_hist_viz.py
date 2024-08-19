"""
Usage: `python3 -m exploration/whole_hist_viz.py path/to/data.tsv figure/save/path.png`

There are three optional arguments after the save path, specifying (in order):
  - A frequency cutoff, such that clades which never exceed the cutoff are not plotted
  - A width, in inches, for the plot
  - A height, in inches, for the plot

A simple plot of the proportions of all currently-named NextStrain clades over
time in the United States, with no per-state breakdown.
"""

import sys

import polars as pl
from plotnine import aes, geom_line, ggplot, theme_bw

from linmod.data import with_bad_ns_assign


def make_plot(dfp, ignore_under):
    r"""
    Plots all NextStrain clades in the available data, weekly, for the full duration.
    """
    df = (
        pl.scan_csv(dfp, separator="\t")
        .rename({"clade_nextstrain": "lineage"})
        .cast({"date": pl.Date, "date_submitted": pl.Date}, strict=False)
        .filter(
            pl.col("lineage").is_not_null(),
            # Drop samples with missing collection or reporting dates
            pl.col("date").is_not_null(),
            # Drop samples claiming to be reported before being collected
            pl.col("date") <= pl.col("date_submitted"),
            country="USA",
            host="Homo sapiens",
        )
    )
    df = with_bad_ns_assign(df, "lineage", "date").filter(
        pl.col("impossible").not_()
    )
    df = (
        df.with_columns(wd=pl.col("date").dt.weekday())
        .with_columns(
            offset=pl.col("wd").map_elements(
                lambda x: "-" + str(x) + "d" if x != 7 else "0d",
                return_dtype=pl.String,
            )
        )
        .with_columns(pl.col("date").dt.offset_by(pl.col("offset")))
        .group_by("lineage", "date")
        .agg(pl.len().alias("count"))
        .with_columns(
            frequency=(pl.col("count") / pl.sum("count").over("date")),
        )
        .select("date", "lineage", "frequency")
        .collect()
    )

    trivial = (
        df.group_by("lineage")
        .agg(pl.col("frequency").max())
        .with_columns(ignorable=(pl.col("frequency") < ignore_under))
    )

    plt = (
        ggplot(
            df.join(
                trivial.drop("frequency"), on="lineage", how="left"
            ).filter(pl.col("ignorable").not_())
        )
        + geom_line(
            aes(
                x="date",
                y="frequency",
                color="lineage",
            ),
        )
        + theme_bw()
    )

    return plt


if __name__ == "__main__":
    # Load configuration, if given
    assert len(sys.argv) >= 3
    print(
        "Visualization routine called with "
        + str(len(sys.argv) - 1)
        + " arguments"
    )

    ignore_under = 0.2
    if len(sys.argv) > 3:
        ignore_under = float(sys.argv[3])

    plt = make_plot(sys.argv[1], ignore_under=ignore_under)

    width = 6
    if len(sys.argv) > 4:
        width = float(sys.argv[4])

    height = 4
    if len(sys.argv) > 5:
        height = float(sys.argv[5])

    plt.save(sys.argv[2], width=width, height=height, limitsize=False)
