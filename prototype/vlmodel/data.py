import lzma
import os
import urllib.request
from pathlib import Path
from typing import List

import polars as pl


def load_metadata(
    url: str,
    columns: List[str],
    filter_expression,
    save_path="metadata-raw.tsv",
    redownload=False,
):
    save_path = Path(save_path)

    # Download the data if necessary
    if redownload or not os.path.exists(save_path):
        print("Downloading data...")
        with urllib.request.urlopen(url) as response:
            with lzma.open(response) as in_file, open(save_path, "wb") as out_file:
                out_file.write(in_file.read())
        print("Download complete.")
    else:
        print("Using cached download.")

    return (
        pl.read_csv(save_path, separator="\t").filter(filter_expression).select(columns)
    )
