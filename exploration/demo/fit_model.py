import argparse
import sys
import jax
import numpy as np
import numpyro
import polars as pl
from numpyro.infer import MCMC, NUTS
import linmod.models


def fit(
    samples,
    model_factory,
    calibration_period: int = 90,
    n_chains: int = 4,
    n_iterations: int = 500,
    n_warmup: int = 2500,
    host_device_count: int = 4,
    forecast_start: int = -30,
    forecast_end: int = 15,
    seed: int = 0,
):
    numpyro.set_host_device_count(host_device_count)

    # load and pre-process data
    data = (
        pl.read_csv(samples)
        .cast({"date": pl.Date}, strict=False)
        .drop_nulls(subset=["date"])  # Drop dates that aren't resolved to the day
        .filter(pl.col("date") >= pl.col("date").max() - calibration_period)
        .select(["lineage", "date", "count", "division"])
        .pivot(on="lineage", index=["date", "division"], values="count")
        .fill_null(0)
    )

    # Extract count matrix, division indices, and time covariate
    counts = data.select(sorted(data.columns)).drop(["date", "division"])
    division_names, divisions = np.unique(data["division"], return_inverse=True)

    model = model_factory(divisions, counts, N=None, num_lineages=None)
    mcmc_model = model.make_mcmc_model()

    mcmc = MCMC(
        NUTS(mcmc_model),
        num_samples=n_iterations,
        num_warmup=n_warmup,
        num_chains=n_chains,
    )

    mcmc.run(
        jax.random.key(seed),
        divisions,
        counts=counts.to_numpy(),
    )

    # Collect posterior regression parameter samples
    return model.postprocess_mcmc(
        mcmc,
        n_chains=n_chains,
        n_iterations=n_iterations,
        division_names=division_names,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("samples", type=argparse.FileType("r"))
    p.add_argument("model", type=str)
    p.add_argument("--output", type=argparse.FileType("w"), default=sys.stdout)

    args = p.parse_args()

    results = fit(
        samples=args.samples,
        model=linmod.models.__dict__[args.model],
    )
    pl.write_csv(results, args.output)
