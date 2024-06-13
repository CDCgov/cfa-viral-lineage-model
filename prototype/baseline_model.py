import arviz as az
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import polars as pl
import vlmodel
from jax.nn import softmax


class BaselineModel(vlmodel.AbstractModel):
    """
    Multinomial with separate probabilities for each clade.
    """

    def preprocess_data(self, data):
        # 1. Re-encode clades as integers from 0.
        #    Clades are ordered lexicographically.
        all_clades = data["clade_nextstrain"].unique().sort()
        num_clades = len(all_clades)

        clades = (
            data["clade_nextstrain"]
            .replace(old=all_clades, new=range(num_clades))
            .cast(pl.Int64)
            .to_numpy()
        )

        return clades, num_clades

    def numpyro_model(self, clades, num_clades):
        z = numpyro.sample("z", dist.Normal(0, 1), sample_shape=(num_clades,))
        numpyro.deterministic("phi", softmax(z))

        # Note that a "logit" is any unnormalized probability to be sent to softmax,
        # not specfically logit(phi)
        numpyro.sample(
            "Y_tl",
            dist.Categorical(logits=z),
            obs=clades,
        )

    def postprocess_samples(self, samples):
        samples.posterior = samples.posterior.drop_vars("z")

        return samples


if __name__ == "__main__":
    data_filter = (
        (pl.col("country") == "USA")
        & pl.col("date").is_not_null()
        & (pl.col("host") == "Homo sapiens")
        & (pl.col("date").str.starts_with("2024-05"))
    )

    # TODO: Eventually, use the full global dataset
    data = vlmodel.load_metadata(
        "https://data.nextstrain.org/files/ncov/open/north-america/metadata.tsv.xz",
        columns=["clade_nextstrain", "date", "division"],
        filter_expression=data_filter,
    )

    model = BaselineModel(data)
    samples = model.sample(1000, 200)

    az.plot_trace(samples)
    plt.show()
