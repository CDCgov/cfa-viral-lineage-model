#!/usr/bin/env python3

import lzma
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import polars as pl

CACHE_DIRECTORY = Path(".cache")


def load_metadata(
    url,
    lineage_column_name,
    redownload=False,
):
    parsed_url = urlparse(url)
    save_path = CACHE_DIRECTORY / parsed_url.netloc / parsed_url.path.lstrip("/")

    # Download the data if necessary
    if redownload or not os.path.exists(save_path):
        with urlopen(url) as response:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with lzma.open(response) as in_file, open(save_path, "wb") as out_file:
                out_file.write(in_file.read())

    return (
        pl.scan_csv(save_path, separator="\t")
        .rename({lineage_column_name: "lineage"})
        .filter(
            (pl.col("country") == "USA")
            & pl.col("date").is_not_null()
            & (pl.col("host") == "Homo sapiens")
        )
        .group_by(["lineage", "date", "division"])
        .agg(pl.len().alias("count"))
        .collect()
    )


if __name__ == "__main__":
    # TODO: Soon, use the full global dataset
    data = load_metadata(
        url="https://data.nextstrain.org/files/ncov/open/north-america/metadata.tsv.xz",
        lineage_column_name="clade_nextstrain",
    )

    print(data.write_csv())
