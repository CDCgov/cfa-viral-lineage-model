# Demo of model fitting and evaluation

Current workflow, from top-level of repo:

1. Download data with `data/load_metadata.py > exploration/demo/metadata.csv`
2. Fit models
    1. `cd exploration/demo`
    2. [If `exploration/demo/out` does not exist] `mkdir out`
    3. `./fit_baseline_model.py metadata.csv > out/samples-baseline.csv`
    4. `./fit_indep_div_model.py metadata.csv > out/samples-id.csv`
3. Evaluate models with `./evaluate.py metadata.csv`
