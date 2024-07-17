#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import polars as pl

from linmod.eval import mae

if len(sys.argv) != 2:
    print(
        "Usage: python3 evaluate.py <data_path>",
        file=sys.stderr,
    )
    sys.exit(1)

data = (
    pl.scan_csv(sys.argv[1])
    .cast({"date": pl.Date}, strict=False)
    .drop_nulls(subset=["date"])  # Drop dates that aren't resolved to the day
    .filter(pl.col("date") >= pl.col("date").max() - 90)
    .with_columns(
        day=(pl.col("date") - pl.max("date")).dt.total_days(),
    )
    .select("lineage", "division", "day", "count")
)

scores = {}

for samples_file in os.listdir("out/"):
    samples_file = Path("out") / samples_file
    samples = pl.scan_csv(samples_file).drop_nulls()
    # TODO: where is the row of nulls coming from

    scores[samples_file.stem] = (
        mae(samples, data).collect().get_column("mae").sum()
    )

for name, score in scores.items():
    print(f"{name}: {score}")
