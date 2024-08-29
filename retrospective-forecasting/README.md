# Model fitting and evaluation

From top-level of repo, run `retrospective-forecasting/main.py retrospective-forecasting/config.yaml`.

This:
1. Creates datasets for forecasting and evaluation (from a cached dataset if available).
2. Runs all forecasting models specified in `retrospective-forecasting/config.yaml` and plots a simple summary.
3. Evaluates the model forecasts using the created evaluation dataset.

Parameters failing MCMC convergence diagnostics are reported, but not excluded from downstream steps.
