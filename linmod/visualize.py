import polars as pl
from plotnine import (
    aes,
    facet_wrap,
    geom_line,
    geom_ribbon,
    ggplot,
    theme_bw,
    ylab,
)


def plot_samples(samples):
    summaries = samples.group_by("division", "lcd_offset", "lineage").agg(
        mean_phi=pl.mean("phi"),
        q_lower=pl.quantile("phi", 0.1),
        q_upper=pl.quantile("phi", 0.9),
    )

    return (
        ggplot(summaries)
        + geom_ribbon(
            aes(
                "lcd_offset",
                ymin="q_lower",
                ymax="q_upper",
                group="lineage",
                fill="lineage",
            ),
            alpha=0.15,
        )
        + geom_line(
            aes("lcd_offset", "mean_phi", group="lineage", color="lineage"),
            size=1.5,
        )
        + facet_wrap("division")
        + theme_bw(base_size=20)
        + ylab("phi")
    )
