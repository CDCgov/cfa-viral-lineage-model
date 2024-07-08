#!/usr/bin/env Rscript

library(tidyverse)

df <- read_csv("out/sim_study.csv")

p <- ggplot(df) +
  geom_boxplot(
    aes(factor(sim_study_iteration), phi_time7),
    fill = "dodgerblue4"
  ) +
  geom_hline(aes(yintercept = true_phi_time7), color = "black", linewidth = 2) +
  facet_wrap(vars(division, lineage)) +
  theme_bw(base_size = 20) +
  expand_limits(y = c(0, 1))

ggsave("out/output.png", p, width = 20, height = 15, dpi = 300)
