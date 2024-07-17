#!/usr/bin/env python3

"""
Usage: `load_metadata.py`

Download the Nextstrain metadata file, preprocess it,
keeping only the divisions specified by the file data/included-divisions.txt,
and print the result to `stdout`.

The output is given in CSV format, with columns `lineage`, `date`, `division`,
and `count`. Rows are uniquely identified by `(lineage, date, division)`.

Preprocessing is done to ensure that:
- Observations without a recorded date are removed;
- Only the 50 U.S. states, D.C., and Puerto Rico are included; and
- Only observations from human hosts are included.

The data is downloaded from:
https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst
"""

import lzma
import os
import sys
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


def load_metadata(
    redownload: bool = False,
    divisions_file: str = "data/included-divisions.txt",
) -> pl.DataFrame:
    """
    Download the metadata file, preprocess it, and return a `polars.DataFrame`.

    The column specified by `LINEAGE_COLUMN_NAME` is renamed to `lineage`.
    The unprocessed (but decompressed) data is cached in the `CACHE_DIRECTORY`.
    If `redownload`, the data is redownloaded, and the cache is replaced.
    """

    parsed_url = urlparse(DATA_SOURCE)
    save_path = (
        CACHE_DIRECTORY
        / parsed_url.netloc
        / parsed_url.path.lstrip("/").rsplit(".", 1)[0]
    )

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

    # Determine which US divisions to include
    with open(divisions_file) as f:
        included_divisions = [line.strip() for line in f]

    # Preprocess the data
    print("Preprocessing data...", file=sys.stderr, flush=True, end="")

    df = (
        pl.scan_csv(save_path, separator="\t")
        .rename({LINEAGE_COLUMN_NAME: "lineage"})
        .filter(
            pl.col("date").is_not_null(),
            pl.col("division").is_in(included_divisions),
            country="USA",
            host="Homo sapiens",
        )
        .group_by("lineage", "date", "division")
        .agg(pl.len().alias("count"))
        .collect()
    )

    print(" done.", file=sys.stderr, flush=True)

    return df


if __name__ == "__main__":
    data = load_metadata()

    print(data.write_csv())
    print("\nSuccess.", file=sys.stderr)
