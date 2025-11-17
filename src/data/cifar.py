import typing

from jax import numpy as jnp
from jax import random
import jaxtyping
from learning.data import datamodule
import tensorflow_datasets as tfds

# Constants
PyTree = jaxtyping.PyTree


class CIFAR10DataModule(datamodule.TFDSDataModule):
    """Data module for loading and preprocessing the CIFAR-10 dataset."""

    def __init__(
        self,
        batch_size: int,
        deterministic: bool = True,
        drop_remainder: bool = False,
        num_workers: int = 4,
        rng: random.KeyArray = random.PRNGKey(42),
        shuffle_buffer_size: int = 1_000,
        preprocess_fn: typing.Optional[datamodule.TF_PREPROCESS_FN] = None,
        *args,
        **kwargs,
    ) -> None:
        """Instantiates the CIFAR-10 data module."""
        self.builder = tfds.builder(name="cifar10", *args, **kwargs)

        super().__init__(
            batch_size=batch_size,
            deterministic=deterministic,
            drop_remainder=drop_remainder,
            num_workers=num_workers,
            rng=rng,
            shuffle_buffer_size=shuffle_buffer_size,
            preprocess_fn=preprocess_fn,
        )

    @property
    def input_features(self) -> PyTree:
        """PyTree: A dictionary with dummy input arrays."""
        size = (1, 32, 32, 3)
        return {"image": jnp.zeros(size, dtype=jnp.float32)}

    @property
    def target_features(self) -> PyTree:
        """PyTree: A dictionary with dummy target arrays."""
        return {"label": jnp.zeros((1,), dtype=jnp.int32)}
