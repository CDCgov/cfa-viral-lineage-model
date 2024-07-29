#!/usr/bin/env python3

import os
from pathlib import Path

import polars as pl

from linmod.visualize import plot_samples

for samples_file in filter(
    lambda path: path.endswith(".csv"),
    os.listdir("out/"),
):
    samples_file = Path("out") / samples_file

    p = plot_samples(pl.read_csv(samples_file))
    p.save(
        f"out/trajectories-{samples_file.stem}.png",
        width=40,
        height=30,
        dpi=300,
        limitsize=False,
    )
