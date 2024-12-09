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

plot_script_file = f"plot_all_{forecast_dir.name}.sh"
with open(plot_script_file, "w") as plot_script:
    plot_script.write("#/usr/bin/sh\n")
    for model_name in config["forecasting"]["models"]:
        print_message(f"Fitting {model_name} model...")
        model_class = getattr(linmod.models, model_name)
        model = model_class(model_data)

        mcmc = MCMC(
            NUTS(model.numpyro_model, dense_mass=model.dense_mass()),
            num_samples=config["forecasting"]["mcmc"]["samples"],
            num_warmup=config["forecasting"]["mcmc"]["warmup"],
            num_chains=config["forecasting"]["mcmc"]["chains"],
            thinning=config["forecasting"]["mcmc"]["thinning"],
        )
        mcmc.run(jax.random.key(0))

        try:
            convergence_config = config["forecasting"]["mcmc"]["convergence"]

            convergence = linmod.models.get_convergence(
                mcmc,
                ignore_nan_in=model.ignore_nan_in(),
                drop_ignorable_nan=False,
            )

            if convergence_config["report_mode"] == "failing":
                convergence = convergence.filter(
                    (pl.col("n_eff") < convergence_config["ess_cutoff"])
                    | (pl.col("r_hat") > convergence_config["psrf_cutoff"])
                )
            convergence.write_parquet(
                forecast_dir / f"convergence_{model_name}.parquet"
            )

            if convergence_config["plot"] and convergence.shape[0] > 0:
                plot_dir = forecast_dir / ("convergence_" + model_name)
                plots = linmod.models.plot_convergence(
                    mcmc, convergence["param"]
                )

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

        forecast.write_parquet(
            forecast_dir / f"forecasts_{model_name}.parquet"
        )

        if config["forecasting"]["survey"]["make"]:
            print_message("Aggregating to HHS regions")
            # Take missing divisions out of map
            hhs_map = {
                div: linmod.data.hhs_regions[div]
                for div in forecast["division"].unique()
            }
            hhs_region_forecast = linmod.models.InfectionWeightedAggregator()(
                forecast,
                hhs_map,
                pop_sizes=pl.read_csv(
                    config["forecasting"]["survey"]["pop_sizes"]
                ),
            )
            agg_forecast_path = (
                forecast_dir / f"hhs_region_forecasts_{model_name}.parquet"
            )
            hhs_region_forecast.write_parquet(agg_forecast_path)

            if config["forecasting"]["survey"]["plot"]:
                png = (
                    forecast_dir
                    / "visualizations"
                    / f"hhs_region_forecasts_{model_name}.png"
                )
                plot_script.write(
                    (
                        "python3 -m linmod.visualize "
                        f"-f {agg_forecast_path} "
                        f"-p {png} "
                        "-t nodata\n"
                    )
                )

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

    for forecast_path in forecast_dir.glob("forecasts_*.parquet"):
        model_name = forecast_path.stem.split("_")[1]
        forecast = pl.scan_parquet(forecast_path)

        for evaluator_config in config["evaluation"]["metrics"]:
            if isinstance(evaluator_config, dict):
                assert (
                    len(evaluator_config) == 1
                ), "Evaluator config is formatted incorrectly."

                evaluator_config = list(evaluator_config.items())
                evaluator_name = evaluator_config[0][0]
                evaluator_args = {
                    k: v for d in evaluator_config[0][1] for k, v in d.items()
                }

            else:
                evaluator_name = evaluator_config
                evaluator_args = {}

            evaluator = getattr(linmod.eval, evaluator_name)(
                samples=forecast,
                data=eval_data.lazy(),
                **evaluator_args,
            )

            for metric_name, metric_function in vars(type(evaluator)).items():
                if metric_name.startswith("_"):
                    continue

                print_message(
                    (
                        f"Evaluating {model_name} model using "
                        f"{evaluator_name}.{metric_name}..."
                    ),
                    end="",
                )

                scores.append(
                    (
                        f"{evaluator_name}.{metric_name}",
                        model_name,
                        metric_function(evaluator),
                    )
                )

                print_message(" done.")

        data_path = config["data"]["save_file"]["eval"]
        png_path = eval_dir / "visualizations" / f"eval_{model_name}.png"
        plot_script.write(
            (
                "python3 -m linmod.visualize "
                f"-f {forecast_path} "
                f"-d {data_path} "
                f"-p {png_path} "
                "-t eval\n"
            )
        )


print_message("Success!")

pl.DataFrame(
    scores,
    schema=["Metric", "Model", "Score"],
    orient="row",
).write_parquet(eval_dir / "results.parquet")
