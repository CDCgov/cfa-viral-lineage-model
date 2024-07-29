#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import polars as pl

from linmod.eval import proportions_energy_score, proportions_mean_norm

score_functions = {
    "mean_norm": proportions_mean_norm,
    "energy_score": proportions_energy_score,
}


if len(sys.argv) != 2:
    print(
        "Usage: python3 evaluate.py <data_path>",
        file=sys.stderr,
    )
    sys.exit(1)

data = pl.scan_csv(sys.argv[1], try_parse_dates=True)

scores = {}

for samples_file in filter(
    lambda path: path.endswith(".csv"),
    os.listdir("out/"),
):
    samples_file = Path("out") / samples_file
    samples = pl.scan_csv(samples_file)

    for score_name, score_func in score_functions.items():
        scores[(samples_file.stem, score_name)] = score_func(samples, data)

for name, score in scores.items():
    print(f"{name}: {score}")
