"""
Usage: `python3 -m linmod.data`

Download the Nextstrain metadata file, preprocess it, and export it.

The output is given in CSV format, with columns `date`, `fd_offset`, `division`,
`lineage`, `count`. Rows are uniquely identified by `(date, division, lineage)`.
`date` and `fd_offset` can be computed from each other, given the forecast date;
the `fd_offset` column is the number of days between the forecast date and the `date`
column.

Preprocessing defaults to the following configuration, which can be overridden
via a YAML file passed as an argument to the script (see `data.DEFAULT_CONFIG`):
- The data is downloaded from
  https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst;
- Only the 90 most recent days of sequences since the forecast date are included,
  where the forecast date defaults to today's date; and
- Only the 50 U.S. states, D.C., and Puerto Rico are included.

Furthermore:
- Observations without a recorded date are removed; and
- Only observations from human hosts are included.
"""

import lzma
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import polars as pl
import yaml
import zstandard

from .utils import print_message

"""
Default configuration for data download, preprocessing, and export.

The configuration dictionary expects all of the following entries in a
`data` key.
"""
DEFAULT_CONFIG = {
    "data": {
        # Where should the data be downloaded from?
        "source": "https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst",
        # Where (directory) should the unprocessed (but decompressed) data be stored?
        "cache_dir": ".cache/",
        # Where (file) should the processed data be stored?
        "save_path": "metadata.csv",
        # Should the data be redownloaded (and the cache replaced)?
        "redownload": False,
        # What column should be renamed to `lineage`?
        "lineage_column_name": "clade_nextstrain",
        # What is the forecast date?
        # No sequences collected after this date are included.
        "forecast_date": {
            "year": datetime.now().year,
            "month": datetime.now().month,
            "day": datetime.now().day,
        },
        # How many days of sequences should be included?
        "num_days": 90,
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

if __name__ == "__main__":
    # Load configuration, if given

    config = DEFAULT_CONFIG

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            config |= yaml.safe_load(f)

    # Download the data, if necessary

    parsed_url = urlparse(config["data"]["source"])
    save_path = (
        Path(config["data"]["cache_dir"])
        / parsed_url.netloc
        / parsed_url.path.lstrip("/").rsplit(".", 1)[0]
    )

    if config["data"]["redownload"] or not os.path.exists(save_path):
        print_message("Downloading...", end="")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with urlopen(config["data"]["source"]) as response, save_path.open(
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

    # Preprocess the data

    print_message("Preprocessing data...", end="")

    forecast_date = pl.date(
        config["data"]["forecast_date"]["year"],
        config["data"]["forecast_date"]["month"],
        config["data"]["forecast_date"]["day"],
    )

    df = (
        pl.scan_csv(save_path, separator="\t")
        .rename({config["data"]["lineage_column_name"]: "lineage"})
        # Cast with `strict=False` replaces invalid values with null,
        # which we can then filter out. Invalid values include dates
        # that are resolved only to the month, not the day
        .cast({"date": pl.Date}, strict=False)
        .filter(
            pl.col("date").is_not_null(),
            pl.col("date") <= forecast_date,
            pl.col("date") >= forecast_date - config["data"]["num_days"],
            pl.col("division").is_in(config["data"]["included_divisions"]),
            country="USA",
            host="Homo sapiens",
        )
        .group_by("lineage", "date", "division")
        .agg(pl.len().alias("count"))
        .with_columns(
            fd_offset=(pl.col("date") - forecast_date).dt.total_days()
        )
        .select("date", "fd_offset", "division", "lineage", "count")
    )

    print_message(" done.")

    # Export data

    print_message("Exporting data...", end="")
    df.collect().write_csv(config["data"]["save_path"])
    print_message(" done.")
