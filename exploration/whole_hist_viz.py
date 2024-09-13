import argparse
import datetime

import numpyro.distributions as dist
import polars as pl
from plotnine import aes, geom_line, geom_ribbon, ggplot, theme_bw


def get_plot_data(
    dfp, first_date, last_date, observed_only, divisions, ci_alpha
):
    r"""
    Compresses history in range to US-aggregate by week.
    """
    if ci_alpha is None:
        z = 0.0
    else:
        z = float(dist.Normal(0, 1).icdf(1.0 - ci_alpha))

    last_samp_date = datetime.date(3000, 1, 1)
    if observed_only:
        last_samp_date = last_date

    df = (
        pl.scan_parquet(dfp, separator="\t")
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
            # If specified, keep only samples seen by last date
            pl.col("date_submitted") <= last_samp_date,
            country="USA",
            host="Homo sapiens",
        )
        .collect()
    )

    if divisions is not None:
        assert all(div in df["division"] for div in divisions)
        df = df.filter(pl.col("division").is_in(divisions))

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
        .with_columns(
            ci_width=(
                z
                / pl.col("count").sqrt()
                * pl.col("frequency").sqrt()
                * (1.0 - pl.col("frequency")).sqrt()
            )
        )
        .with_columns(upper_ci=(pl.col("frequency") + pl.col("ci_width")))
        .with_columns(lower_ci=(pl.col("frequency") - pl.col("ci_width")))
        .select("date", "lineage", "frequency", "lower_ci", "upper_ci")
    )

    return df


def make_plot(
    dfp,
    ignore_under,
    first_date,
    last_date,
    observed_only,
    divisions,
    ci_alpha,
):
    r"""
    Plots all NextStrain clades in the available data, weekly, for the full duration.
    """
    df = get_plot_data(
        dfp, first_date, last_date, observed_only, divisions, ci_alpha
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

    if ci_alpha is not None:
        plt = plt + geom_ribbon(
            aes(
                x="date",
                ymin="lower_ci",
                ymax="upper_ci",
                fill="lineage",
            ),
            alpha=0.15,
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
        "-d",
        "--division",
        default=None,
        help="Plot will only include data for these divisions. Default corresponds to all divisions.",
    )
    parser.add_argument(
        "-c",
        "--frequency_cutoff",
        default=0.5,
        help="Plot will only include lineages which exceed this frequency in at least one week",
    )
    parser.add_argument(
        "-u",
        "--uncertainty_alpha",
        default=None,
        help="If specified, plot will include weekly uncertainty in proportion via Wald intervals at this alpha.",
    )
    parser.add_argument(
        "--observed_only",
        action="store_true",
        help="Controls whether only sequences observed before the sampling date are visualized.",
    )
    parser.add_argument(
        "-w", "--width", default=6, help="Plot width, in inches."
    )
    parser.add_argument(
        "-t", "--height", default=4, help="Plot height, in inches."
    )

    args = parser.parse_args()

    divisions = None
    if args.division is not None:
        divisions = args.division.strip().split(",")

    plt = make_plot(
        args.data_filename,
        ignore_under=float(args.frequency_cutoff),
        first_date=datetime.datetime.strptime(args.first_date, "%Y-%m-%d"),
        last_date=datetime.datetime.strptime(args.last_date, "%Y-%m-%d"),
        observed_only=args.observed_only,
        divisions=divisions,
        ci_alpha=float(args.uncertainty_alpha),
    )

    plt.save(
        args.output_filepath,
        width=float(args.width),
        height=float(args.height),
        limitsize=False,
    )
