import sys

import chex
import jax.numpy as jnp
import pytest

from learning.core import mixin as _mixin
from learning.data import cifar


@pytest.mark.parametrize("batch_size", [1, 4])
def test_cifar_10_datamodule(batch_size: int) -> None:
    """Test the CIFAR-10 datamodule."""
    dm = cifar.CIFAR10DataModule(batch_size=batch_size)
    assert isinstance(dm, _mixin.DataMixin)

    data = next(dm.train_dataloader())
    assert isinstance(data, dict)
    assert "image" in data
    chex.assert_shape(data["image"], (batch_size, 32, 32, 3))
    chex.assert_type(data["image"], jnp.uint8)  # CIFAR-10 images are uint8
    assert "label" in data
    chex.assert_shape(data["label"], (batch_size,))
    chex.assert_type(data["label"], jnp.int32)

    data = next(dm.val_dataloader())
    assert isinstance(data, dict)
    assert "image" in data
    chex.assert_shape(data["image"], (batch_size, 32, 32, 3))
    chex.assert_type(data["image"], jnp.uint8)
    assert "label" in data
    chex.assert_shape(data["label"], (batch_size,))
    chex.assert_type(data["label"], jnp.int32)

    data = next(dm.test_dataloader())
    assert isinstance(data, dict)
    assert "image" in data
    chex.assert_shape(data["image"], (batch_size, 32, 32, 3))
    chex.assert_type(data["image"], jnp.uint8)
    assert "label" in data
    chex.assert_shape(data["label"], (batch_size,))
    chex.assert_type(data["label"], jnp.int32)

    assert isinstance(dm.input_features, dict)
    assert "image" in dm.input_features
    chex.assert_shape(dm.input_features["image"], (1, 32, 32, 3))
    chex.assert_type(dm.input_features["image"], jnp.float32)
    assert isinstance(dm.target_features, dict)
    assert "label" in dm.target_features
    chex.assert_shape(dm.target_features["label"], (1,))
    chex.assert_type(dm.target_features["label"], jnp.int32)


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
