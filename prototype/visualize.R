#!/usr/bin/env Rscript

library(tidyverse)

samples <- read_csv("output.csv") |>
	pivot_longer(
		starts_with("phi"),
		names_to = "t",
		names_prefix = "phi_t",
		values_to = "phi"
	) |>
	mutate(
		t = as.factor(as.numeric(t))
	) |>
	# TODO: why are there NAs
	drop_na()

max_a_posteriori <- samples |>
	group_by(division, t) |>
	slice_max(potential_energy, n = 1) |>
	rename(map_phi = phi) |>
	select(division, lineage, t, map_phi)

quantiles <- samples |>
	group_by(division, t, lineage) |>
	summarize(
		lower = quantile(phi, 0.1),
		upper = quantile(phi, 0.9)
	)

full_join(max_a_posteriori, quantiles, by = c("division", "t", "lineage")) |>
	filter(str_starts(division, "A")) |>
	ggplot() +
	geom_ribbon(
		aes(t, ymin = lower, ymax = upper, group = lineage, fill = lineage),
		alpha = 0.15
	) +
	geom_line(
		aes(t, map_phi, group = lineage, color = lineage),
		linewidth = 1.5
	) +
	facet_wrap(vars(division)) +
	theme_bw(base_size = 20) +
	ylab(expression(phi))
