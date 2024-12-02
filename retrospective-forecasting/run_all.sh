#!/bin/sh

echo "Running and evaluating models for all configs"
for conf in `ls config/`; do ./main.py config/$conf; done

echo "Plotting results for all models for all configs"
for plotsh in `ls plot_all_*`; do sh $plotsh; done
rm plot_all_*
