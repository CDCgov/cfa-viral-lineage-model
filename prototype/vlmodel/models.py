from random import randint
from typing import List

import arviz as az
import jax.numpy as jnp
import polars as pl
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS


class AbstractModel:
    def __init__(self, data: pl.DataFrame, **hyperparameters):
        self.data = self.preprocess_data(data)
        self.hyperparameters = hyperparameters

    def sample(
        self,
        num_samples: int,
        num_warmup: int,
        seed: int = None,
    ) -> az.InferenceData:
        seed = seed if seed is not None else randint(0, 10000)

        mcmc = MCMC(
            NUTS(self.numpyro_model),
            num_samples=num_samples,
            num_warmup=num_warmup,
        )

        mcmc.run(
            PRNGKey(seed),
            *self.data,
            **self.hyperparameters,
        )

        return self.postprocess_samples(az.from_numpyro(mcmc))

    def preprocess_data(self, data: pl.DataFrame) -> List[jnp.ndarray]:
        """
        Define any dataframe preprocessing here.
        Return a list of `numpy`-type arrays, which will be given to the numpyro model.
        """

        raise NotImplementedError

    def numpyro_model(self, data: List[jnp.ndarray], **hyperparameters):
        """
        Define the numpyro model here.
        Assume `data` is a list of `numpy`-type arrays output from `preprocess_data`.
        """

        raise NotImplementedError

    def postprocess_samples(self, samples: az.InferenceData) -> az.InferenceData:
        """
        Define any postprocessing of the samples here, if desired.
        Return the samples data structure.
        """

        return samples
