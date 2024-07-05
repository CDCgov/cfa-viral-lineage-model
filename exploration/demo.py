import numpy as np
import numpyro
import numpyro.distributions as dist
import jax
import polars as pl
import pandas as pd
import altair as alt


def my_dist(p: float, total_count: int):
    """Observation model"""
    print(f"{p=}")
    return dist.Binomial(total_count=total_count, probs=p)


def my_model(n_successes, n_total):
    p = numpyro.sample("p", dist.Beta(0.5, 0.5))
    numpyro.sample("obs", my_dist(p, n_total), obs=n_successes)


# Get simulated data ------------------------------------------------------------------------------
key = jax.random.key(1234)
true_p = 0.75
n_totals = [10, 100, 1000]
n_sims = 5


def run_sim(key, n_total):
    sim_dist = my_dist(p=true_p, total_count=n_total)

    sample_key, run_key = jax.random.split(key, 2)
    sim_data = sim_dist.sample(sample_key)

    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(my_model),
        num_samples=500,
        num_warmup=500,
    )

    mcmc.run(run_key, n_successes=sim_data, n_total=n_total)

    return np.array(mcmc.get_samples()["p"])


results = pl.concat(
    [
        pl.DataFrame({"sim": sim, "n_total": n_total, "p": run_sim(subkey, n_total)})
        for n_total in n_totals
        for sim, subkey in enumerate(jax.random.split(key, n_sims))
    ]
)

# boxplot of results across simulations, with true value shown as a line
alt.data_transformers.disable_max_rows()
base_chart = alt.Chart(results.with_columns(true=true_p).to_pandas())
line = base_chart.mark_rule().encode(y="true")
boxplot = base_chart.mark_boxplot().encode(x="sim:N", y="p")
(line + boxplot).facet("n_total")
