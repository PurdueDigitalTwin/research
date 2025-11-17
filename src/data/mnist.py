import typing

from jax import numpy as jnp
from jax import random
import jaxtyping
import tensorflow as tf
import tensorflow_datasets as tfds

from learning.data import datamodule

# Type Aliases
TF_FN = typing.Callable[
    [typing.Dict[str, tf.Tensor], typing.Any],
    typing.Dict[str, tf.Tensor],
]
PyTree = jaxtyping.PyTree


class MNISTDataModule(datamodule.TFDSDataModule):
    """Data module for loading and preprocessing the MNIST dataset."""

    def __init__(
        self,
        *,
        batch_size: int,
        deterministic: bool = True,
        drop_remainder: bool = False,
        num_workers: int = 4,
        rng: random.KeyArray = random.PRNGKey(42),
        shuffle_buffer_size: int = 1_000,
        preprocess_fn: typing.Optional[TF_FN] = None,
        **kwargs,
    ) -> None:
        """Instantiates the MNIST data module."""
        self.builder = tfds.builder("mnist", **kwargs)

        super().__init__(
            batch_size=batch_size,
            deterministic=deterministic,
            drop_remainder=drop_remainder,
            num_workers=num_workers,
            rng=rng,
            shuffle_buffer_size=shuffle_buffer_size,
            preprocess_fn=preprocess_fn,
            **kwargs,
        )

    @property
    def input_features(self) -> PyTree:
        """PyTree: A dictionary with dummy input arrays."""
        size = (1, 28, 28, 1)
        return {"image": jnp.zeros(size, dtype=jnp.float32)}

    @property
    def target_features(self) -> PyTree:
        """PyTree: A dictionary with dummy target arrays."""
        return {"label": jnp.zeros((1,), dtype=jnp.int32)}
