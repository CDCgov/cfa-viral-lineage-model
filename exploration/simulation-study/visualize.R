#!/usr/bin/env Rscript

library(tidyverse)

file_path <- commandArgs(trailingOnly = TRUE)[1]
df <- read_csv(file_path)

p <- ggplot(df) +
  geom_boxplot(
    aes(factor(sim_study_iteration), phi_present),
    fill = "dodgerblue4"
  ) +
  geom_hline(
    aes(yintercept = true_phi_present),
    color = "black", linewidth = 2
  ) +
  facet_grid(rows = vars(division), cols = vars(lineage)) +
  theme_bw(base_size = 20) +
  expand_limits(y = c(0, 1)) +
  theme(axis.text.x = element_blank())

ggsave("output.png", p, width = 20, height = 15, dpi = 300)
