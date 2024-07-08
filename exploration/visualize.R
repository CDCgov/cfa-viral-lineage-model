#!/usr/bin/env Rscript

library(tidyverse)

samples <- read_csv("out/output.csv") |>
  pivot_longer(
    starts_with("phi"),
    names_to = "t",
    names_prefix = "phi_t",
    values_to = "phi"
  ) |>
  mutate(
    t = str_replace(t, "m", "-") |> as.numeric()
  ) |>
  # TODO: why are there NAs
  drop_na()

max_a_posteriori <- samples |>
  group_by(division, t) |>
  slice_max(potential_energy, n = 1) |>
  rename(map_phi = phi) |>
  select(division, lineage, t, map_phi)

means <- samples |>
  group_by(division, t, lineage) |>
  summarize(
    mean_phi = mean(phi)
  )

quantiles <- samples |>
  group_by(division, t, lineage) |>
  summarize(
    q_lower = quantile(phi, 0.1),
    q_upper = quantile(phi, 0.9)
  )

summaries <- means |>
  full_join(quantiles, by = c("division", "t", "lineage")) |>
  full_join(max_a_posteriori, by = c("division", "t", "lineage"))

p <- summaries |>
  ggplot() +
  geom_ribbon(
    aes(t, ymin = q_lower, ymax = q_upper, group = lineage, fill = lineage),
    alpha = 0.15
  ) +
  geom_line(
    aes(t, mean_phi, group = lineage, color = lineage),
    linewidth = 1.5
  ) +
  facet_wrap(vars(division)) +
  theme_bw(base_size = 20) +
  ylab(expression(phi))

ggsave(p, "out/output.png", width = 40, height = 30, dpi = 300)
