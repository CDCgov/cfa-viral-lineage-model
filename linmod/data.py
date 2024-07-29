"""
Usage: `python3 -m linmod.data`

Download the Nextstrain metadata file, preprocess it, and print the result to `stdout`.

The output is given in CSV format, with columns `date`, `lcd_offset`, `division`,
`lineage`, `count`. Rows are uniquely identified by `(date, division, lineage)`.
`date` and `lcd_offset` can be computed from each other, given the forecast date;
the `lcd_offset` column is the number of days between the last collection date
(defaults to today) and the `date` column.

Preprocessing is done to ensure that:
- The most recent 90 days of sequences are included;
- Observations without a recorded date are removed;
- Only the 50 U.S. states, D.C., and Puerto Rico are included; and
- Only observations from human hosts are included.

The data is downloaded from:
https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst
"""

import lzma
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import polars as pl
import zstandard

# Configuration

# Where should the unprocessed (but decompressed) data be stored?
CACHE_DIRECTORY = Path(".cache")

# Where should the data be downloaded from?
DATA_SOURCE = "https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst"

# What column should be renamed to `lineage`?
LINEAGE_COLUMN_NAME = "clade_nextstrain"

# How many days of sequences should be included?
NUM_DAYS = 90

# Which divisions should be included?
# Currently set to the 50 U.S. states, D.C., and Puerto Rico
INCLUDED_DIVISIONS = [
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
]


def load_metadata(
    last_collection_date: tuple | None = None,
    redownload: bool = False,
) -> pl.DataFrame:
    """
    Download the metadata file, preprocess it, and return a `polars.DataFrame`.

    The data is filtered to include only the most recent `NUM_DAYS` days of sequences
    collected by `last_collection_date`, specified as a tuple `(year, month, day)`
    (defaulting to today's date if not specified). The column specified by
    `LINEAGE_COLUMN_NAME` is renamed to `lineage`. The unprocessed (but decompressed)
    data is cached in the `CACHE_DIRECTORY`. If `redownload`, the data is redownloaded,
    and the cache is replaced.
    """

    if last_collection_date is None:
        now = datetime.now()
        last_collection_date = (now.year, now.month, now.day)

    parsed_url = urlparse(DATA_SOURCE)
    save_path = (
        CACHE_DIRECTORY
        / parsed_url.netloc
        / parsed_url.path.lstrip("/").rsplit(".", 1)[0]
    )
    # TODO: should cache save path incorporate `last_collection_date`?

    # Download the data if necessary
    if redownload or not os.path.exists(save_path):
        print("Downloading...", file=sys.stderr, flush=True, end="")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with urlopen(DATA_SOURCE) as response, save_path.open(
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

        print(" done.", file=sys.stderr, flush=True)
    else:
        print("Using cached data.", file=sys.stderr, flush=True)

    # Preprocess the data
    print("Preprocessing data...", file=sys.stderr, flush=True, end="")

    df = (
        pl.scan_csv(save_path, separator="\t")
        .rename({LINEAGE_COLUMN_NAME: "lineage"})
        # Cast with `strict=False` replaces invalid values with null,
        # which we can then filter out. Invalid values include dates
        # that are resolved only to the month, not the day
        .cast({"date": pl.Date}, strict=False)
        .filter(
            pl.col("date").is_not_null(),
            pl.col("date") <= pl.date(*last_collection_date),
            pl.col("division").is_in(INCLUDED_DIVISIONS),
            country="USA",
            host="Homo sapiens",
        )
        .filter(
            pl.col("date") >= pl.col("date").max() - 90,
        )
        .group_by("lineage", "date", "division")
        .agg(pl.len().alias("count"))
        .with_columns(
            lcd_offset=(
                pl.col("date") - pl.date(*last_collection_date)
            ).dt.total_days()
        )
        .select("date", "lcd_offset", "division", "lineage", "count")
    )

    print(" done.", file=sys.stderr, flush=True)

    return df


if __name__ == "__main__":
    data = load_metadata()
    # TODO: an argparse setup to specify `last_collection_date`

    print(data.collect().write_csv(), end="")
    print("\nSuccess.", file=sys.stderr)
