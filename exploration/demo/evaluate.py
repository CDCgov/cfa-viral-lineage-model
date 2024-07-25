#!/usr/bin/env python3

import os
from pathlib import Path
import yaml
import polars as pl
import argparse

import linmod.eval


def score(data_path, model_names, score_names, samples_dir):
    data = (
        pl.scan_csv(data_path)
        .cast({"date": pl.Date}, strict=False)
        .drop_nulls(subset=["date"])  # Drop dates that aren't resolved to the day
        .filter(pl.col("date") >= pl.col("date").max() - 90)
        .with_columns(
            day=(pl.col("date") - pl.max("date")).dt.total_days(),
        )
        .select("lineage", "division", "day", "count")
    )

    for model_name in model_names:
        model = linmod.models.__dict__[model_name]
        # this is where you would run the model, or check that the output exists

    scores = {}

    for samples_file in os.listdir(samples_dir):
        samples_file = Path(samples_dir) / samples_file
        samples = pl.scan_csv(samples_file).drop_nulls()
        # TODO: where is the row of nulls coming from

        for score_name in score_names:
            score_func = linmod.eval.__dict__[score_name]
            scores[(samples_file.stem, score_name)] = score_func(samples, data)

    for name, score in scores.items():
        print(f"{name}: {score}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data", type=argparse.FileType("r"))
    p.add_argument("config", type=argparse.FileType("r"))
    p.add_argument("samples_dir", default="out/")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    score(
        data_path=args.data,
        model_names=config["models"],
        score_names=config["scores"],
        samples_dir=args.samples_dir,
    )
