"""
Usage: `python3 -m linmod.data [path/to/config.yaml]`

Download the Nextstrain metadata file, preprocess it, and export it.

Two datasets are exported: one for model fitting and one for evaluation.
The model dataset contains sequences collected and reported by a specified
forecast date, while the evaluation dataset extends the horizon into the future.

To change default behaviors, create a yaml configuration file with the key ["data"],
and pass it in the call to this script. For a list of configurable sub-keys, see the
`DEFAULT_CONFIG` dictionary.

The output is given in CSV format, with columns `date`, `fd_offset`, `division`,
`lineage`, `count`. Rows are uniquely identified by `(date, division, lineage)`.
`date` and `fd_offset` can be computed from each other, given the forecast date;
the `fd_offset` column is the number of days between the forecast date and the `date`
column, such that, for example, 0 is the forecast date, -1 the day before, and 1 the
day after.

Note that observations without a recorded date are removed, and only observations
from human hosts are included.
"""

import lzma
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen

import polars as pl
import yaml
import zstandard

from .utils import print_message

DEFAULT_CONFIG = {
    "data": {
        # Where should the data be downloaded from?
        "source": "https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst",
        # Where (directory) should the unprocessed (but decompressed) data be stored?
        "cache_dir": ".cache/",
        # Where (files) should the processed datasets for modeling and evaluation
        # be stored?
        "save_file": {
            "model": "data/metadata-model.csv",
            "eval": "data/metadata-eval.csv",
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
    }
}
"""
Default configuration for data download, preprocessing, and export.

The configuration dictionary expects all of the following entries in a
`data` key.
"""


def main(cfg: Optional[dict]):
    config = DEFAULT_CONFIG

    if cfg is not None:
        config["data"] |= cfg["data"]

    # Download the data, if necessary

    parsed_url = urlparse(config["data"]["source"])
    cache_path = (
        Path(config["data"]["cache_dir"])
        / parsed_url.netloc
        / parsed_url.path.lstrip("/").rsplit(".", 1)[0]
    )

    if config["data"]["redownload"] or not cache_path.exists():
        print_message("Downloading...", end="")

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with urlopen(config["data"]["source"]) as response, cache_path.open(
            "wb"
        ) as out_file:
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
        f'{config["data"]["horizon"]["lower"]}d'
    )
    horizon_upper_date = forecast_date.dt.offset_by(
        f'{config["data"]["horizon"]["upper"]}d'
    )

    lineage_year = (
        pl.col("lineage")
        .replace("recombinant", "0")
        .str.extract(r"(\d+)")
        .str.to_integer(strict=True)
        .add(2000)
    )

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
            # Drop impossible lineage assigments
            # (lineages which had not yet been named, e.g. a sequence
            # in 2020 cannot belong to 23D)
            pl.col("date").dt.year() >= lineage_year,
            # Drop samples not from humans in the included US divisions
            pl.col("division").is_in(config["data"]["included_divisions"]),
            country="USA",
            host="Homo sapiens",
        )
    )

    eval_df = (
        full_df.group_by("lineage", "date", "division")
        .agg(pl.len().alias("count"))
        .with_columns(
            fd_offset=(pl.col("date") - forecast_date).dt.total_days()
        )
        .select("date", "fd_offset", "division", "lineage", "count")
        .collect()
    )

    Path(config["data"]["save_file"]["eval"]).parent.mkdir(
        parents=True, exist_ok=True
    )

    eval_df.write_csv(config["data"]["save_file"]["eval"])

    print_message(" done.")
    print_message("Exporting modeling dataset...", end="")

    model_df = (
        full_df.filter(pl.col("date_submitted") <= forecast_date)
        .group_by("lineage", "date", "division")
        .agg(pl.len().alias("count"))
        .with_columns(
            fd_offset=(pl.col("date") - forecast_date).dt.total_days()
        )
        .select("date", "fd_offset", "division", "lineage", "count")
        .collect()
    )

    Path(config["data"]["save_file"]["model"]).parent.mkdir(
        parents=True, exist_ok=True
    )

    model_df.write_csv(config["data"]["save_file"]["model"])

    print_message(" done.")


if __name__ == "__main__":
    # Load configuration, if given

    cfg = None
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            cfg = yaml.safe_load(f)["data"]

    main(cfg)
