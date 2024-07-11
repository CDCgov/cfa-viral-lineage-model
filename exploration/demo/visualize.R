#!/usr/bin/env Rscript

library(tidyverse)

file_path <- commandArgs(trailingOnly = TRUE)[1]

samples <- read_csv(file_path) |>
  pivot_longer(
    starts_with("phi"),
    names_to = "t",
    names_prefix = "phi_t",
    values_to = "phi"
  ) |>
  mutate(
    t = str_replace(t, "m", "-") |> as.numeric()
  )

summaries <- samples |>
  group_by(division, t, lineage) |>
  summarize(
    mean_phi = mean(phi),
    q_lower = quantile(phi, 0.1),
    q_upper = quantile(phi, 0.9)
  )

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

ggsave("output.png", p, width = 40, height = 30, dpi = 300)
