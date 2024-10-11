#!/usr/bin/env python3

import argparse
import shutil

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

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "config",
    type=str,
    help="Path to YAML configuration file",
)
yaml_path = parser.parse_args().config

with open(yaml_path) as f:
    config = yaml.safe_load(f)

# Create the datasets
linmod.data.main(config)

# Load the dataset used for retrospective forecasting

model_data = pl.read_parquet(config["data"]["save_file"]["model"])

# Fit each model

forecast_dir = ValidPath(config["forecasting"]["save_dir"])

if forecast_dir.exists():
    print_message("Removing existing output directory and all contents.")
    shutil.rmtree(forecast_dir)

forecast_dir.mkdir()

for model_name in config["forecasting"]["models"]:
    print_message(f"Fitting {model_name} model...")
    model_class = linmod.models.__dict__[model_name]
    model = model_class(model_data)

    mcmc = MCMC(
        NUTS(model.numpyro_model, dense_mass=True),
        num_samples=config["forecasting"]["mcmc"]["samples"],
        num_warmup=config["forecasting"]["mcmc"]["warmup"],
        num_chains=config["forecasting"]["mcmc"]["chains"],
    )
    mcmc.run(jax.random.key(0))

    try:
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
                    < config["forecasting"]["mcmc"]["convergence"][
                        "ess_cutoff"
                    ]
                )
                | (
                    pl.col("r_hat")
                    > config["forecasting"]["mcmc"]["convergence"][
                        "psrf_cutoff"
                    ]
                )
            )
        convergence.write_parquet(
            forecast_dir / f"convergence_{model_name}.parquet"
        )

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

        del convergence

    except Exception as e:
        print_message("An error occurred while checking convergence:")
        print_message(e)

    forecast = model.create_forecasts(
        mcmc,
        np.arange(
            config["data"]["horizon"]["lower"],
            config["data"]["horizon"]["upper"] + 1,
        ),
    )

    forecast.write_parquet(forecast_dir / f"forecasts_{model_name}.parquet")

    print_message("Done.")

    # Try to free up some memory
    del model, mcmc, forecast


# Load the full evaluation dataset

eval_data = pl.read_parquet(config["data"]["save_file"]["eval"])

viz_data = eval_data.filter(
    pl.col("lineage").is_in(model_data["lineage"].unique()),
    pl.col("fd_offset") > 0,
)

# Evaluate each model
eval_dir = ValidPath(config["evaluation"]["save_dir"])

scores = []

for metric_name in config["evaluation"]["metrics"]:
    metric_function = linmod.eval.__dict__[metric_name]

    for forecast_path in forecast_dir.glob("forecasts_*.parquet"):
        model_name = forecast_path.stem.split("_")[1]
        print_message(
            f"Evaluating {model_name} model using {metric_name}...", end=""
        )

        forecast = pl.scan_parquet(forecast_path)
        scores.append(
            (
                metric_name,
                model_name,
                metric_function(forecast, eval_data.lazy()),
            )
        )

        plot_forecast(forecast.collect(), viz_data).save(
            eval_dir / "visualizations" / f"eval_{model_name}.png",
            width=25,
            height=15,
            dpi=200,
            verbose=False,
        )

        print_message(" done.")

print_message("Success!")

pl.DataFrame(
    scores,
    schema=["Metric", "Model", "Score"],
    orient="row",
).write_parquet(eval_dir / "results.parquet")
