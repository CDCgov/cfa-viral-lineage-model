import argparse
import datetime

import numpyro.distributions as dist
import polars as pl
from plotnine import aes, geom_line, geom_ribbon, ggplot, theme_bw

import linmod.data


def get_plot_data(
    ns_path,
    usher_path,
    clades_as_of,
    first_date,
    last_date,
    observed_only,
    divisions,
    lineages,
    ci_alpha,
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

    mal = True if (lineages is None or len(lineages) == 0) else False

    df = linmod.data.process_nextstrain(
        ns_path,
        rename={"clade_nextstrain": "lineage"},
        horizon_lower_date=first_date,
        horizon_upper_date=last_date,
        included_divisions=linmod.data.DEFAULT_CONFIG["data"][
            "included_divisions"
        ],
        model_all_lineages=mal,
        included_lineages=lineages,
    ).filter(
        # If specified, keep only samples seen by last date
        pl.col("date_submitted")
        <= last_samp_date,
    )

    if usher_path:
        df = linmod.data.recode_clades_using_usher(
            df,
            usher_path,
            usher_lineage_from=linmod.data.DEFAULT_CONFIG["data"][
                "usher_lineage_column_name"
            ],
        )

    if clades_as_of is not None:
        df = linmod.data.combine_clades(df, clades_as_of)

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
    ns_path,
    usher_path,
    clades_as_of,
    ignore_under,
    first_date,
    last_date,
    observed_only,
    divisions,
    lineages,
    ci_alpha,
):
    r"""
    Plots all NextStrain clades in the available data, weekly, for the full duration.
    """
    df = get_plot_data(
        ns_path,
        usher_path,
        clades_as_of,
        first_date,
        last_date,
        observed_only,
        divisions,
        lineages,
        ci_alpha,
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
    parser.add_argument(
        "--clades_as_of",
        default=None,
        help="If specified, we use cladecombiner.AsOfAggregator to aggregate to clades from this date.",
    )
    parser.add_argument(
        "--usher_path",
        default="",
        help="Path to UShER data. If provided, clade calls are taken from this, not Nextstrain (main) data.",
    )
    parser.add_argument(
        "--lineages",
        default=None,
        help="Plot will only include data for these lineages. Default corresponds to all lineages.",
    )

    args = parser.parse_args()

    divisions = None
    if args.division is not None:
        divisions = args.division.strip().split(",")

    lineages = None
    if args.lineages is not None:
        lineages = args.lineages.strip().split(",")

    cao = (
        None
        if args.clades_as_of is None
        else datetime.datetime.strptime(args.clades_as_of, "%Y-%m-%d").date()
    )

    plt = make_plot(
        args.data_filename,
        args.usher_path,
        cao,
        ignore_under=float(args.frequency_cutoff),
        first_date=datetime.datetime.strptime(args.first_date, "%Y-%m-%d"),
        last_date=datetime.datetime.strptime(args.last_date, "%Y-%m-%d"),
        observed_only=args.observed_only,
        divisions=divisions,
        lineages=lineages,
        ci_alpha=float(args.uncertainty_alpha),
    )

    plt.save(
        args.output_filepath,
        width=float(args.width),
        height=float(args.height),
        limitsize=False,
    )
