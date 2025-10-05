import functools
import typing

from absl import logging
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import tensorflow as tf
from tensorflow_datasets import core as tfds_core

from learning.core import mixin as _mixin

# Constants
NP_JAX_DTYPE_MAP = {
    np.uint8: jnp.uint8,
    np.int32: jnp.int32,
    np.int64: jnp.int32,  # map int64 to int32
    np.float32: jnp.float32,
    np.float64: jnp.float32,  # map float64 to float32
    np.bool_: jnp.bool_,
}
PyTree = jaxtyping.PyTree


# Type Aliases
TF_PREPROCESS_FN = typing.Callable[
    [typing.Dict[str, tf.Tensor], typing.Any],
    typing.Dict[str, tf.Tensor],
]


# ==============================================================================
# Helper function
# ==============================================================================
def convert_to_jax_array(example: PyTree) -> PyTree:
    """Converts a PyTree of numpy arrays to JAX arrays."""
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if x.dtype in NP_JAX_DTYPE_MAP else x,
        example,
    )


class TFDSDataModule(_mixin.DataMixin):
    """Base class for data module using Tensorflow Datasets (TFDS).

    Attributes:
        batch_size (int): Batch size for data loading.
        builder (tfds_core.DatasetBuilder): TFDS dataset builder instance.
        deterministic (bool): Whether to use deterministic data loading.
        drop_remainder (bool): Whether to drop the last incomplete batch.
        num_workers (int): Number of parallel workers for data loading.
        rng (jax.random.KeyArray): JAX random key for shuffling.
        preprocess_fn (Optional[TF_PREPROCESS_FN]): Optional preprocessing
            function for the tensorflow dataset.
        shuffle_buffer_size (int): Buffer size for shuffling the dataset.
    """

    builder: tfds_core.DatasetBuilder
    """tfds.core.DatasetBuilder: TFDS dataset builder instance."""

    def __init__(
        self,
        batch_size: int,
        deterministic: bool,
        drop_remainder: bool,
        num_workers: int,
        rng: jax.random.KeyArray,
        preprocess_fn: typing.Optional[TF_PREPROCESS_FN],
        shuffle_buffer_size: int,
    ) -> None:
        """Instantiate a `TFDSDataModule`."""
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.drop_remainder = drop_remainder
        self.num_workers = num_workers
        self.rng = rng
        self.preprocess_fn = preprocess_fn
        self.shuffle_buffer_size = shuffle_buffer_size

        self.prepare_data()

    def prepare_data(self) -> None:
        """Downloads and prepares the dataset."""
        self.builder.download_and_prepare()

    def train_dataloader(self) -> typing.Generator[PyTree, None, None]:
        """Returns the training dataloader."""
        self.rng, local_rng = jax.random.split(self.rng, num=2)
        seed = local_rng[0].item()  # type: ignore[arg-type]
        ds = self.prepare_sharded_dataset(
            split="train",
            shuffle_files=True,
            shuffle_seed=seed,
        )
        for batch in ds.as_numpy_iterator():
            yield convert_to_jax_array(batch)

    def val_dataloader(self) -> typing.Generator[PyTree, None, None]:
        """Returns the validation dataloader."""
        if "validation" not in self.builder.info.splits:
            logging.warning(
                "Dataset does not have a validation split. "
                "Using the test split for validation."
            )
            ds = self.prepare_sharded_dataset(
                split="test",
                shuffle_files=False,
            )
        else:
            ds = self.prepare_sharded_dataset(
                split="validation",
                shuffle_files=False,
            )
        for batch in ds.as_numpy_iterator():
            yield convert_to_jax_array(batch)

    def test_dataloader(self) -> typing.Generator[PyTree, None, None]:
        """Returns the test dataloader."""
        ds = self.prepare_sharded_dataset(
            split="test",
            shuffle_files=False,
        )
        for batch in ds.as_numpy_iterator():
            yield convert_to_jax_array(batch)

    def prepare_sharded_dataset(
        self,
        split: str,
        shuffle_files: bool,
        shuffle_seed: typing.Optional[int] = None,
    ) -> tf.data.Dataset:
        """Returns a sharded `tensorflow.data.Dataset` for distributed loading.

        Args:
            split (str): Dataset split to load.
            shuffle_files (bool): Whether to shuffle the files before loading.
            shuffle_seed (Optional[int], optional): Random seed for shuffling.

        Returns:
            tf.data.Dataset: A sharded TensorFlow dataset.
        """
        if split not in self.builder.info.splits:
            raise ValueError(
                f"Invalid split {split:s}: "
                f"Expected one of {list(self.builder.info.splits.keys())}."
            )

        ds = self.builder.as_dataset(
            split=split,
            as_supervised=False,  # NOTE: intentionally for returning a dict
            shuffle_files=shuffle_files,
        )
        assert isinstance(ds, tf.data.Dataset), "Failed to load dataset."

        def _make_sharded_dataset(
            shard_index: int,
            num_shards: int,
            dataset: tf.data.Dataset,
        ) -> tf.data.Dataset:
            """Function to create a single sharded dataset."""
            local_dataset = dataset.shard(
                index=shard_index,
                num_shards=num_shards,
            )

            # shuffle the dataset
            if shuffle_seed is not None:
                # NOTE: shuffle the dataset if a seed is given
                local_seed = jax.random.fold_in(
                    jax.random.PRNGKey(shuffle_seed),
                    jax.process_index(),
                )[0]
                local_dataset = local_dataset.shuffle(
                    buffer_size=self.shuffle_buffer_size,
                    seed=local_seed,
                )

            # apply optional preprocessing pipeline
            if self.preprocess_fn is not None:
                local_dataset = local_dataset.map(
                    map_func=self.preprocess_fn,
                    deterministic=self.deterministic,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
            local_dataset = local_dataset.batch(
                batch_size=self.batch_size,
                deterministic=self.deterministic,
                drop_remainder=self.drop_remainder,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

            return local_dataset

        indices = tf.data.Dataset.range(self.num_workers)
        dataset = indices.interleave(
            map_func=functools.partial(
                _make_sharded_dataset, num_shards=self.num_workers, dataset=ds
            ),
            deterministic=self.deterministic,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
