"""
Download the Nextstrain metadata file, preprocess it, and export it.

Two datasets are exported: one for model fitting and one for evaluation.
The model dataset contains sequences collected and reported by a specified
forecast date, while the evaluation dataset extends the horizon into the future.

To change default behaviors, create a yaml configuration file with the key ["data"],
and pass it in the call to this script. For a list of configurable sub-keys, see the
`DEFAULT_CONFIG` dictionary.

The output is given in Apache Parquet format, with columns `date`, `fd_offset`,
`division`, `lineage`, `count`. Rows are uniquely identified by
`(date, division, lineage)`. `date` and `fd_offset` can be computed from each other,
given the forecast date; the `fd_offset` column is the number of days between the
forecast date and the `date` column, such that, for example, 0 is the forecast date,
-1 the day before, and 1 the day after.

Note that observations without a recorded date are removed, and only observations
from human hosts are included.
"""

import argparse
import lzma
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen

import polars as pl
import yaml
import zstandard

from .utils import ValidPath, expand_grid, print_message

DEFAULT_CONFIG = {
    "data": {
        # Where should the data be downloaded from?
        "source": "https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst",
        # Where (directory) should the unprocessed (but decompressed) data be stored?
        "cache_dir": ".cache/",
        # Where (files) should the processed datasets for modeling and evaluation
        # be stored?
        "save_file": {
            "model": "data/metadata-model.parquet",
            "eval": "data/metadata-eval.parquet",
        },
        # Should the data be redownloaded (and the cache replaced)?
        "redownload": False,
        # What column should be renamed to `lineage`?
        "lineage_column_name": "clade_nextstrain",
        # What is the forecast date?
        # No sequences collected or reported after this date are included in the
        # modeling dataset.
        "forecast_date": {
            "year": datetime.now().year,
            "month": datetime.now().month,
            "day": datetime.now().day,
        },
        # How many days since the forecast date should be included in the datasets?
        # The evaluation dataset will contain sequences collected and reported within
        # this horizon. The modeling dataset will contain sequences collected and
        # reported within the horizon `[lower, 0]`.
        "horizon": {
            "lower": -90,
            "upper": 14,
        },
        # Which divisions should be included?
        # Currently set to the 50 U.S. states, D.C., and Puerto Rico
        "included_divisions": [
            "Alabama",
            "Alaska",
            "Arizona",
            "Arkansas",
            "California",
            "Colorado",
            "Connecticut",
            "Delaware",
            "Florida",
            "Georgia",
            "Hawaii",
            "Idaho",
            "Illinois",
            "Indiana",
            "Iowa",
            "Kansas",
            "Kentucky",
            "Louisiana",
            "Maine",
            "Maryland",
            "Massachusetts",
            "Michigan",
            "Minnesota",
            "Mississippi",
            "Missouri",
            "Montana",
            "Nebraska",
            "Nevada",
            "New Hampshire",
            "New Jersey",
            "New Mexico",
            "New York",
            "North Carolina",
            "North Dakota",
            "Ohio",
            "Oklahoma",
            "Oregon",
            "Pennsylvania",
            "Puerto Rico",
            "Rhode Island",
            "South Carolina",
            "South Dakota",
            "Tennessee",
            "Texas",
            "Utah",
            "Vermont",
            "Virginia",
            "Washington",
            "Washington DC",
            "West Virginia",
            "Wisconsin",
            "Wyoming",
        ],
        # Which lineages should be included?
        # If not provided, all observed lineages are included.
        # If a list of length >= 1, all observed lineages not in this list are grouped
        # into "other".
        "lineages": [],
    }
}
"""
Default configuration for data download, preprocessing, and export.

The configuration dictionary expects all of the following entries in a
`data` key.
"""

hhs_regions = {
    "Connecticut": "HHS-1",
    "Maine": "HHS-1",
    "Massachusetts": "HHS-1",
    "New Hampshire": "HHS-1",
    "Rhode Island": "HHS-1",
    "Vermont": "HHS-1",
    "New Jersey": "HHS-2",
    "New York": "HHS-2",
    "Puerto Rico": "HHS-2",
    "Virgin Islands": "HHS-2",
    "Delaware": "HHS-3",
    "Washington DC": "HHS-3",
    "Maryland": "HHS-3",
    "Pennsylvania": "HHS-3",
    "Virginia": "HHS-3",
    "West Virginia": "HHS-3",
    "Alabama": "HHS-4",
    "Florida": "HHS-4",
    "Georgia": "HHS-4",
    "Kentucky": "HHS-4",
    "Mississippi": "HHS-4",
    "North Carolina": "HHS-4",
    "South Carolina": "HHS-4",
    "Tennessee": "HHS-4",
    "Illinois": "HHS-5",
    "Indiana": "HHS-5",
    "Michigan": "HHS-5",
    "Minnesota": "HHS-5",
    "Ohio": "HHS-5",
    "Wisconsin": "HHS-5",
    "Arkansas": "HHS-6",
    "Louisiana": "HHS-6",
    "New Mexico": "HHS-6",
    "Oklahoma": "HHS-6",
    "Texas": "HHS-6",
    "Iowa": "HHS-7",
    "Kansas": "HHS-7",
    "Missouri": "HHS-7",
    "Nebraska": "HHS-7",
    "Colorado": "HHS-8",
    "Montana": "HHS-8",
    "North Dakota": "HHS-8",
    "South Dakota": "HHS-8",
    "Utah": "HHS-8",
    "Wyoming": "HHS-8",
    "Arizona": "HHS-9",
    "California": "HHS-9",
    "Hawaii": "HHS-9",
    "Nevada": "HHS-9",
    "American Samoa": "HHS-9",
    "Commonwealth of the Northern Mariana Islands": "HHS-9",
    "Federated States of Micronesia": "HHS-9",
    "Guam": "HHS-9",
    "Marshall Islands": "HHS-9",
    "Republic of Palau": "HHS-9",
    "Alaska": "HHS-10",
    "Idaho": "HHS-10",
    "Oregon": "HHS-10",
    "Washington": "HHS-10",
}
"""
Dictionary form of https://www.hhs.gov/about/agencies/iea/regional-offices/index.html, except that DC is Washington DC not District of Columbia
"""


class CountsFrame(pl.DataFrame):
    """
    A `polars.DataFrame` which enforces a format for observed counts of lineages.

    See `REQUIRED_COLUMNS` for the expected columns.
    """

    REQUIRED_COLUMNS = {
        "date",
        "fd_offset",
        "division",
        "lineage",
        "count",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.validate()

    @classmethod
    def read_parquet(cls, *args, **kwargs):
        return cls(pl.read_parquet(*args, **kwargs))

    def validate(self):
        assert self.REQUIRED_COLUMNS.issubset(
            self.columns
        ), f"Missing required columns: ({', '.join(self.REQUIRED_COLUMNS - set(self.columns))})"

        assert (
            self.null_count().sum_horizontal().item() == 0
        ), "Null values detected in the dataset."

        assert self[
            "count"
        ].dtype.is_integer(), "Count column must be an integer type."


def main(cfg: Optional[dict]):
    config = DEFAULT_CONFIG

    if cfg is not None:
        config["data"] |= cfg["data"]

    # Download the data, if necessary

    parsed_url = urlparse(config["data"]["source"])
    cache_path = (
        ValidPath(config["data"]["cache_dir"])
        / parsed_url.netloc
        / parsed_url.path.lstrip("/").rsplit(".", 1)[0]
    )

    if config["data"]["redownload"] or not cache_path.exists():
        print_message("Downloading...", end="")

        with (
            urlopen(config["data"]["source"]) as response,
            cache_path.open("wb") as out_file,
        ):
            if parsed_url.path.endswith(".gz"):
                with lzma.open(response) as in_file:
                    out_file.write(in_file.read())

            elif parsed_url.path.endswith(".zst"):
                decompressor = zstandard.ZstdDecompressor()

                with decompressor.stream_reader(response) as reader:
                    out_file.write(reader.readall())

            else:
                raise ValueError(f"Unsupported file format: {parsed_url.path}")

        print_message(" done.")
    else:
        print_message("Using cached data.")

    # Preprocess and export the data

    print_message("Exporting evaluation dataset...", end="")

    forecast_date = pl.date(
        config["data"]["forecast_date"]["year"],
        config["data"]["forecast_date"]["month"],
        config["data"]["forecast_date"]["day"],
    )

    horizon_lower_date = forecast_date.dt.offset_by(
        f"{config['data']['horizon']['lower']}d"
    )
    horizon_upper_date = forecast_date.dt.offset_by(
        f"{config['data']['horizon']['upper']}d"
    )

    model_all_lineages = len(config["data"]["lineages"]) == 0

    full_df = (
        pl.scan_csv(cache_path, separator="\t")
        .rename({config["data"]["lineage_column_name"]: "lineage"})
        # Cast with `strict=False` replaces invalid values with null,
        # which we can then filter out. Invalid values include dates
        # that are resolved only to the month, not the day
        .cast({"date": pl.Date, "date_submitted": pl.Date}, strict=False)
        .filter(
            # Drop samples with missing lineage
            pl.col("lineage").is_not_null(),
            # Drop samples with missing collection or reporting dates
            pl.col("date").is_not_null(),
            pl.col("date_submitted").is_not_null(),
            # Drop samples collected outside the horizon
            horizon_lower_date <= pl.col("date"),
            pl.col("date") <= horizon_upper_date,
            # Drop samples claiming to be reported before being collected
            pl.col("date") <= pl.col("date_submitted"),
            # Drop samples not from humans in the included US divisions
            pl.col("division").is_in(config["data"]["included_divisions"]),
            country="USA",
            host="Homo sapiens",
        )
        .with_columns(
            lineage=pl.when(
                pl.col("lineage").is_in(config["data"]["lineages"])
                | model_all_lineages
            )
            .then(pl.col("lineage"))
            .otherwise(pl.lit("other"))
        )
        .collect()
    )

    # Generate every combination of date-division-lineage, so that:
    #  1. The evaluation dataset will be evaluation-ready, with 0 counts
    #     where applicable
    #  2. The modeling dataset will have every lineage of interest represented,
    #     even if a lineage was only sampled in the evaluation period
    observations_key = expand_grid(
        date=full_df["date"].unique(),
        division=full_df["division"].unique(),
        lineage=full_df["lineage"].unique(),
    )

    eval_df = CountsFrame(
        full_df.group_by("lineage", "date", "division")
        .agg(count=pl.len())
        .join(
            observations_key,
            on=("date", "division", "lineage"),
            how="right",
        )
        .with_columns(
            fd_offset=(pl.col("date") - forecast_date).dt.total_days(),
            count=pl.col("count").fill_null(0),
        )
        .select("date", "fd_offset", "division", "lineage", "count")
        # Sort to guarantee consistent output, since `.unique()` does not
        .sort("fd_offset", "division", "lineage")
    )

    eval_df.write_parquet(ValidPath(config["data"]["save_file"]["eval"]))

    print_message(" done.")
    print_message("Exporting modeling dataset...", end="")

    model_df = CountsFrame(
        full_df.filter(pl.col("date_submitted") <= forecast_date)
        .group_by("lineage", "date", "division")
        .agg(count=pl.len())
        .join(
            observations_key,
            on=("date", "division", "lineage"),
            how="right",
        )
        .with_columns(
            fd_offset=(pl.col("date") - forecast_date).dt.total_days(),
            count=pl.col("count").fill_null(0),
        )
        .select("date", "fd_offset", "division", "lineage", "count")
        # Remove division-days where no samples were collected, for brevity
        .filter(pl.sum("count").over("date", "division") > 0)
        # Sort to guarantee consistent output, since `.unique()` does not
        .sort("fd_offset", "division", "lineage")
    )

    model_df.write_parquet(ValidPath(config["data"]["save_file"]["model"]))

    print_message(" done.")

    if model_all_lineages:
        print_message(
            "Modeling all lineages observed in the data at any point in the horizon."
        )
    else:
        print_message(
            (
                "Modeling the following subset of lineages, "
                '(all other lineages grouped into "other"): '
            )
            + str(config["data"]["lineages"])
        )


if __name__ == "__main__":
    # Load configuration, if given

    parser = argparse.ArgumentParser(
        prog="python3 -m linmod.data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    yaml_path = parser.parse_args().config

    cfg = None
    if yaml_path is not None:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

    main(cfg)
