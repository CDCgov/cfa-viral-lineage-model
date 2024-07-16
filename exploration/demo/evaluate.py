#!/usr/bin/env python3

from pathlib import Path

import polars as pl

from linmod.eval.proportions import mae

data = (
    pl.scan_csv(Path("../../data/metadata.csv"))
    .cast({"date": pl.Date}, strict=False)
    .drop_nulls(subset=["date"])  # Drop dates that aren't resolved to the day
    .filter(pl.col("date") >= pl.col("date").max() - 90)
    .with_columns(
        day=(pl.col("date") - pl.max("date")).dt.total_days(),
    )
    .select("lineage", "division", "day", "count")
)

scores = {}

for samples_file in ("out/samples-baseline.csv", "out/samples-id.csv"):
    name = Path(samples_file).stem
    samples = pl.scan_csv(samples_file).drop_nulls()
    # TODO: where is the row of nulls coming from

    scores[name] = mae(samples, data).collect().get_column("mae").sum()

for name, score in scores.items():
    print(f"{name}: {score}")
