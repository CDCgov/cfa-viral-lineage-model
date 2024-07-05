library(tidyverse)

medians <- read_csv("medians.csv")

p <- ggplot(medians, aes(phi_time7)) +
	geom_histogram() +
	facet_wrap(vars(division, lineage), scales = "free") +
	geom_vline(aes(xintercept = true_phi_time7))

ggsave("output.png", p, width = 40, height = 30, dpi = 300)