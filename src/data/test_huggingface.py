import sys
import typing

import numpy as np
import pytest
import tensorflow as tf

from src.data import huggingface


def _default_transform(
    image: tf.Tensor,
    label: tf.Tensor,
) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    r"""A default transform function for testing."""
    new_image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    assert isinstance(new_image, tf.Tensor)
    return new_image, label


def test_imagenet1k_datamodule() -> None:
    r"""Test the `ImageNet1KDataModule` implementation."""

    # test instantiation
    dm = huggingface.ImageNet1KDataModule(
        batch_size=2,
        num_workers=1,
        seed=0,
        transform=_default_transform,
        streaming=False,
    )
    assert dm.batch_size == 2
    assert dm.deterministic is True
    assert dm.drop_remainder is True
    assert dm.num_workers == 1
    assert dm.num_train_examples == 1_281_167
    assert dm.num_val_examples == 50_000
    assert dm.num_test_examples == 100_000
    assert dm.seed == 0
    assert all(key in dm.splits for key in ["train", "validation", "test"])

    # test training dataloader
    image, label = next(iter(dm.train_dataloader()))
    np.testing.assert_equal(image.shape, (2, 224, 224, 3))
    np.testing.assert_array_less(label, 1000)

    # test evaluation dataloader
    image, label = next(iter(dm.eval_dataloader()))
    np.testing.assert_equal(image.shape, (2, 224, 224, 3))
    np.testing.assert_array_less(label, 1000)

    # test testing dataloader
    image, label = next(iter(dm.test_dataloader()))
    np.testing.assert_equal(image.shape, (2, 224, 224, 3))
    np.testing.assert_array_less(label, 1000)


def test_mnist_datamodule() -> None:
    r"""Test the `MNISTDataModule` implementation."""

    # test instantiation
    dm = huggingface.MNISTDataModule(
        batch_size=2,
        num_workers=1,
        seed=0,
        transform=_default_transform,
        streaming=False,
    )
    assert dm.batch_size == 2
    assert dm.deterministic is True
    assert dm.drop_remainder is True
    assert dm.num_workers == 1
    assert dm.num_train_examples == 60000
    assert dm.num_val_examples == 10000
    assert dm.num_test_examples == 10000
    assert dm.seed == 0
    assert all(key in dm.splits for key in ["train", "test"])

    # test training dataloader
    image, label = next(iter(dm.train_dataloader()))
    np.testing.assert_equal(image.shape, (2, 224, 224, 3))
    np.testing.assert_array_less(label, 10)

    # test evaluation dataloader
    image, label = next(iter(dm.eval_dataloader()))
    np.testing.assert_equal(image.shape, (2, 224, 224, 3))
    np.testing.assert_array_less(label, 10)

    # test testing dataloader
    image, label = next(iter(dm.test_dataloader()))
    np.testing.assert_equal(image.shape, (2, 224, 224, 3))
    np.testing.assert_array_less(label, 10)


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
