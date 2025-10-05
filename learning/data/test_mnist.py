import sys

import chex
import jax.numpy as jnp
import pytest

from learning.core import mixin as _mixin
from learning.data import mnist


@pytest.mark.parametrize("batch_size", [1, 4])
def test_mnist_datamodule(batch_size: int) -> None:
    """Test the MNIST datamodule."""
    ds = mnist.MNISTDataModule(batch_size=batch_size)
    assert isinstance(ds, _mixin.DataMixin)

    data = next(ds.train_dataloader())
    assert isinstance(data, dict)
    assert "image" in data
    chex.assert_shape(data["image"], (batch_size, 28, 28, 1))
    chex.assert_type(data["image"], jnp.uint8)  # MNIST images are uint8
    assert "label" in data
    chex.assert_shape(data["label"], (batch_size,))
    chex.assert_type(data["label"], jnp.int32)

    data = next(ds.val_dataloader())
    assert isinstance(data, dict)
    assert "image" in data
    chex.assert_shape(data["image"], (batch_size, 28, 28, 1))
    chex.assert_type(data["image"], jnp.uint8)  # MNIST images are uint8
    assert "label" in data
    chex.assert_shape(data["label"], (batch_size,))
    chex.assert_type(data["label"], jnp.int32)

    data = next(ds.test_dataloader())
    assert isinstance(data, dict)
    assert "image" in data
    chex.assert_shape(data["image"], (batch_size, 28, 28, 1))
    chex.assert_type(data["image"], jnp.uint8)  # MNIST images are uint8
    assert "label" in data
    chex.assert_shape(data["label"], (batch_size,))
    chex.assert_type(data["label"], jnp.int32)

    assert isinstance(ds.input_features, dict)
    assert "image" in ds.input_features
    chex.assert_shape(ds.input_features["image"], (1, 28, 28, 1))
    chex.assert_type(ds.input_features["image"], jnp.float32)
    assert isinstance(ds.target_features, dict)
    assert "label" in ds.target_features
    chex.assert_shape(ds.target_features["label"], (1,))
    chex.assert_type(ds.target_features["label"], jnp.int32)


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
