# Model fitting and evaluation

To run analyses:
```
cd retrospective-forecasting
./run_all.sh
```

For each `.yaml` file in `retrospective-forecasting/config`, this:
1. Creates datasets for forecasting and evaluation (from a cached dataset if available).
2. Runs all forecasting models specified in the config and plots a simple summary.
3. Evaluates the model forecasts using the created evaluation dataset.

Outputs are split between `out/eval` and `out/forecasts` and nested therein in subfolders named for the config files.

Parameters failing MCMC convergence diagnostics are reported, but not excluded from downstream steps.

Changes to analyses are made on a config by config basis.

# Data provenance
The population size data in are the 2023 estimates from [NST-EST2023-POP](https://www2.census.gov/programs-surveys/popest/tables/2020-2023/state/totals/NST-EST2023-POP.xlsx]).
