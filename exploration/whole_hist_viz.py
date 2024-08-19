import argparse
import datetime

import polars as pl
from plotnine import aes, geom_line, ggplot, theme_bw

from linmod.data import with_bad_ns_assign


def make_plot(dfp, ignore_under, first_date, last_date):
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
            # Keep only samples in range of specified days
            pl.col("date") >= first_date,
            pl.col("date") <= last_date,
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
    parser = argparse.ArgumentParser(
        description="A simple plot of the proportions of all currently-named NextStrain clades over time in the United States, with no per-state breakdown."
    )
    parser.add_argument("data_filename")
    parser.add_argument("output_filepath")
    parser.add_argument(
        "-f",
        "--first_date",
        default="1900-01-01",
        help="Plot will only include data for this day or later. Default corresponds to all days.",
    )
    parser.add_argument(
        "-l",
        "--last_date",
        default="2100-01-01",
        help="Plot will only include data for up to this day or earlier. Default corresponds to all days.",
    )
    parser.add_argument(
        "-c",
        "--frequency_cutoff",
        default=0.5,
        help="Plot will only include lineages which exceed this frequency in at least one week",
    )
    parser.add_argument(
        "-w", "--width", default=6, help="Plot width, in inches."
    )
    parser.add_argument(
        "-t", "--height", default=4, help="Plot height, in inches."
    )

    args = parser.parse_args()

    plt = make_plot(
        args.data_filename,
        ignore_under=float(args.frequency_cutoff),
        first_date=datetime.datetime.strptime(args.first_date, "%Y-%m-%d"),
        last_date=datetime.datetime.strptime(args.last_date, "%Y-%m-%d"),
    )

    plt.save(
        args.output_filepath,
        width=float(args.width),
        height=float(args.height),
        limitsize=False,
    )
