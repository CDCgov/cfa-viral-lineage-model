# Model fitting and evaluation

This folder contains a retrospective forecasting pipeline, complete with config files setup for retrospective analyses of 2022-02-14, 2023-06-01, and 2024-10-01.
By default, when run, it will download all data needed from [Nextstrain](https://docs.nextstrain.org/projects/ncov/en/latest/reference/remote_inputs.html) and [UShER](https://hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2) and create a local `.cache` folder to house these.
Vintaged data is available compressed from <https://zenodo.org/records/16942110> (if using vintaged data, download the archive, uncompress it, resulting in a folder called `.cache`, and copy the entire `.cache` folder into the top level of `retrospective-forecasting/`)


To run the retrospective analyses, you will first want to make sure that the package is ready to go and all dependencies are available.
```
poetry install
```
Then
```
cd retrospective-forecasting
poetry run ./run_all.sh
```

## Details and customization

For each `.yaml` file in `retrospective-forecasting/config`, the pipeline:
1. Creates datasets for forecasting and evaluation (from a cached dataset if available).
2. Runs all forecasting models specified in the config and plots a simple summary.
3. Evaluates the model forecasts using the created evaluation dataset.

Outputs are split between `out/eval` and `out/forecasts` and nested therein in subfolders named for the config files.

Parameters failing MCMC convergence diagnostics are reported, but not excluded from downstream steps.

Changes to analyses are made on a config by config basis.
That is, if you change the MCMC settings in only one config, it applies only to the analysis of that forecast date (useful, e.g., if one date has slow-mixing or slow-converging MCMC).
Note that the names of the configs themselves are arbitrary: the retrospective forecast date is determined by the `["data"]["forecast_date"]` setting, and the output names by several configurable keys.

# Data provenance
The population size data in are the 2023 estimates from [NST-EST2023-POP](https://www2.census.gov/programs-surveys/popest/tables/2020-2023/state/totals/NST-EST2023-POP.xlsx]).

Vintaged Nextstrain and UShER data is available compressed from <https://zenodo.org/records/16942110>.
