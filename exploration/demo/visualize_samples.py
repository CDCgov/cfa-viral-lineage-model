import os
from pathlib import Path

import polars as pl

from linmod.eval import plot_samples

for samples_file in os.listdir("out/"):
    name = Path(samples_file).stem
    samples = pl.scan_csv(samples_file).drop_nulls()
    # TODO: where is the row of nulls coming from

    p = plot_samples(samples.collect())
    p.save(
        f"out/trajectories-{name}.png",
        width=40,
        height=30,
        dpi=300,
        limitsize=False,
    )
