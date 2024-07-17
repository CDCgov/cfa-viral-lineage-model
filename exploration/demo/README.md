# Demo of model fitting and evaluation

Current workflow:

1. Download data with `../../data/load_metadata.py > metadata.csv`
2. Fit models with `./fit_baseline_model.py metadata.csv > out/samples-baseline.csv` and `./fit_indep_div_model.py metadata.csv > out/samples-id.csv`
3. Evaluate models with `./evaluate.py metadata.csv`
