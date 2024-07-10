#!/usr/bin/env python3

import lzma
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import polars as pl
import zstandard

CACHE_DIRECTORY = Path(".cache")


def load_metadata(
    url,
    lineage_column_name,
    redownload=False,
):
    parsed_url = urlparse(url)
    save_path = (
        CACHE_DIRECTORY / parsed_url.netloc / parsed_url.path.lstrip("/")
    )
    # TODO: the save_path preserves the compression extension,
    # but I export the uncompressed file

    # Download the data if necessary
    if redownload or not os.path.exists(save_path):
        with urlopen(url) as response:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if parsed_url.path.endswith(".gz"):
                with lzma.open(response) as in_file, open(
                    save_path, "wb"
                ) as out_file:
                    out_file.write(in_file.read())
            elif parsed_url.path.endswith(".zst"):
                with zstandard.ZstdDecompressor().stream_reader(
                    response
                ) as reader, open(save_path, "wb") as out_file:
                    out_file.write(reader.readall())
            else:
                raise ValueError(f"Unsupported file format: {parsed_url.path}")

    return (
        pl.scan_csv(save_path, separator="\t")
        .rename({lineage_column_name: "lineage"})
        .filter(
            pl.col("date").is_not_null(),
            country="USA",
            host="Homo sapiens",
        )
        .group_by("lineage", "date", "division")
        .agg(pl.len().alias("count"))
        .collect()
    )


if __name__ == "__main__":
    data = load_metadata(
        url="https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst",
        lineage_column_name="clade_nextstrain",
    )

    print(data.write_csv())
