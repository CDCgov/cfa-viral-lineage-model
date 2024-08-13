#!/usr/bin/env python3

import sys
from pathlib import Path

import jax
import numpy as np
import numpyro
import polars as pl
from numpyro.infer import MCMC, NUTS

import linmod.models as models
from linmod.eval import proportions_energy_score, proportions_mean_norm

numpyro.set_host_device_count(4)


def log(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)
    sys.stderr.flush()


# Configuration

MODELS = {
    "baseline": models.BaselineModel,
    "indep_div": models.IndependentDivisionsModel,
}

METRICS = {
    "mean_norm": proportions_mean_norm,
    "energy_score": proportions_energy_score,
}

FORECAST_DIR = Path("out")

# Load the data

if len(sys.argv) != 2:
    log("Usage: python3 main.py <data_path>")
    sys.exit(1)

data = pl.read_csv(sys.argv[1], try_parse_dates=True)

# Fit each model

FORECAST_DIR.mkdir(exist_ok=True)

for model_name, model_class in MODELS.items():

    forecast_path = FORECAST_DIR / f"forecasts-{model_name}.csv"

    if forecast_path.exists():
        log(f"{model_name} fit already exists; reusing forecast.")
        continue

    log(f"Fitting {model_name} model...")
    model = model_class(data)

    mcmc = MCMC(
        NUTS(model.numpyro_model),
        num_samples=500,
        num_warmup=2500,
        num_chains=4,
    )
    mcmc.run(jax.random.key(0))

    model.create_forecasts(mcmc, np.arange(-30, 15)).write_csv(forecast_path)

    log("Done.")

# Evaluate each model

scores = []

for score_name, score_function in METRICS.items():
    for forecast_path in FORECAST_DIR.glob("forecasts-*.csv"):
        model_name = forecast_path.stem.split("-")[1]
        log(f"Evaluating {model_name} model using {score_name}...", end="")

        forecast = pl.scan_csv(forecast_path)
        scores.append(
            (score_name, model_name, score_function(forecast, data.lazy()))
        )

        log(" done.")

log("Success!")

print(
    pl.DataFrame(
        scores,
        schema=["Metric", "Model", "Score"],
        orient="row",
    ).write_csv(),
    end="",
)
