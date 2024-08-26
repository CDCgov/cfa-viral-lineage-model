#!/usr/bin/env python3

import shutil
import sys

import jax
import numpy as np
import numpyro
import polars as pl
import yaml
from numpyro.infer import MCMC, NUTS

import linmod.data
import linmod.eval
import linmod.models
from linmod.utils import ValidPath, print_message
from linmod.visualize import plot_forecast

numpyro.set_host_device_count(4)


# Load configuration

if len(sys.argv) != 2:
    print_message("Usage: python3 main.py <YAML config path>")
    sys.exit(1)

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

# Create the datasets
linmod.data.main(config)

# Load the dataset used for retrospective forecasting

model_data = pl.read_csv(
    config["data"]["save_file"]["model"], try_parse_dates=True
)

# Fit each model

forecast_dir = ValidPath(config["forecasting"]["save_dir"])

if forecast_dir.exists():
    print_message("Removing existing output directory and all contents.")
    shutil.rmtree(forecast_dir)

forecast_dir.mkdir()

for model_name in config["forecasting"]["models"]:
    forecast_path = forecast_dir / f"forecasts_{model_name}.csv"

    print_message(f"Fitting {model_name} model...")
    model_class = linmod.models.__dict__[model_name]
    model = model_class(model_data)

    mcmc = MCMC(
        NUTS(model.numpyro_model),
        num_samples=config["forecasting"]["mcmc"]["samples"],
        num_warmup=config["forecasting"]["mcmc"]["warmup"],
        num_chains=config["forecasting"]["mcmc"]["chains"],
    )
    mcmc.run(jax.random.key(0))

    convergence = linmod.models.get_convergence(
        mcmc,
        ignore_nan_in=config["forecasting"]["mcmc"]["convergence"][
            "ignore_nan_in"
        ],
    )

    if (
        config["forecasting"]["mcmc"]["convergence"]["report_mode"]
        == "failing"
    ):
        convergence = convergence.filter(
            (
                pl.col("n_eff")
                < config["forecasting"]["mcmc"]["convergence"]["ess_cutoff"]
            )
            | (
                pl.col("r_hat")
                > config["forecasting"]["mcmc"]["convergence"]["psrf_cutoff"]
            )
        )
    convergence.write_csv(forecast_dir / f"convergence_{model_name}.csv")

    if (
        config["forecasting"]["mcmc"]["convergence"]["plot"]
        and convergence.shape[0] > 0
    ):
        plot_dir = forecast_dir / ("convergence_" + model_name)
        plots = linmod.models.plot_convergence(mcmc, convergence["param"])
        for plot, par in zip(plots, convergence["param"].to_list()):
            plot.save(plot_dir / (par + ".png"), verbose=False)

        # Try to free up some memory
        del plots, plot

    forecast = model.create_forecasts(
        mcmc,
        np.arange(
            config["data"]["horizon"]["lower"],
            config["data"]["horizon"]["upper"] + 1,
        ),
    )

    forecast.write_csv(forecast_path)

    plot_forecast(forecast, model_data).save(
        forecast_dir / "visualizations" / f"forecasts_{model_name}.png",
        width=40,
        height=30,
        dpi=300,
        limitsize=False,
    )

    # Try to free up some memory
    del model, mcmc, convergence, forecast

    print_message("Done.")

# Load the full evaluation dataset

eval_data = pl.read_csv(
    config["data"]["save_file"]["eval"], try_parse_dates=True
)

viz_data = eval_data.filter(
    pl.col("lineage").is_in(model_data["lineage"].unique()),
    pl.col("fd_offset") > 0,
)

# Evaluate each model
eval_dir = ValidPath(config["evaluation"]["save_dir"])

scores = []

for metric_name in config["evaluation"]["metrics"]:
    metric_function = linmod.eval.__dict__[metric_name]

    for forecast_path in forecast_dir.glob("forecasts_*.csv"):
        model_name = forecast_path.stem.split("_")[1]
        print_message(
            f"Evaluating {model_name} model using {metric_name}...", end=""
        )

        forecast = pl.scan_csv(forecast_path)
        scores.append(
            (
                metric_name,
                model_name,
                metric_function(forecast, eval_data.lazy()),
            )
        )

        plot_forecast(forecast.collect(), viz_data).save(
            eval_dir / "visualizations" / f"eval_{model_name}.png",
            width=40,
            height=30,
            dpi=300,
            limitsize=False,
        )

        print_message(" done.")

print_message("Success!")

pl.DataFrame(
    scores,
    schema=["Metric", "Model", "Score"],
    orient="row",
).write_csv(eval_dir / "results.csv")
