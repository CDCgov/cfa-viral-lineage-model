#!/usr/bin/env python3

import sys
from pathlib import Path

import jax
import numpy as np
import numpyro
import polars as pl
import yaml
from numpyro.infer import MCMC, NUTS

import linmod.eval
import linmod.models

numpyro.set_host_device_count(4)


def log(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)
    sys.stderr.flush()


# Load configuration

if len(sys.argv) != 2:
    log("Usage: python3 main.py <YAML config path>")
    sys.exit(1)

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

# Load the data

data = pl.read_csv(config["paths"]["data"], try_parse_dates=True)

# Fit each model

forecast_dir = Path(config["paths"]["forecast_dir"])
forecast_dir.mkdir(exist_ok=True)

for model_name in config["models"]:
    forecast_path = forecast_dir / f"forecasts-{model_name}.csv"

    if forecast_path.exists():
        log(f"{model_name} fit already exists; reusing forecast.")
        continue

    log(f"Fitting {model_name} model...")
    model_class = linmod.models.__dict__[model_name]
    model = model_class(data)

    mcmc = MCMC(
        NUTS(model.numpyro_model),
        num_samples=config["mcmc"]["samples"],
        num_warmup=config["mcmc"]["warmup"],
        num_chains=config["mcmc"]["chains"],
    )
    mcmc.run(jax.random.key(0))

    model.create_forecasts(
        mcmc,
        np.arange(
            config["forecast_horizon"]["lower"],
            config["forecast_horizon"]["upper"] + 1,
        ),
    ).write_csv(forecast_path)

    log("Done.")

# Evaluate each model

scores = []

for metric_name in config["metrics"]:
    metric_function = linmod.eval.__dict__[metric_name]

    for forecast_path in forecast_dir.glob("forecasts-*.csv"):
        model_name = forecast_path.stem.split("-")[1]
        log(f"Evaluating {model_name} model using {metric_name}...", end="")

        forecast = pl.scan_csv(forecast_path)
        scores.append(
            (metric_name, model_name, metric_function(forecast, data.lazy()))
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
