import sys

from learning.data import preprocess
import numpy as np
import pytest
import tensorflow as tf


def test_chain() -> None:
    """Test the `chain` function for sequential preprocessing."""

    def add_one(x: int) -> int:
        return x + 1

    def multiply_by_two(x: int) -> int:
        return x * 2

    # create a chained preprocessing function
    chained_fn = preprocess.chain(add_one, multiply_by_two)

    # test the chained function
    input_value = 3
    expected_output = (input_value + 1) * 2  # (3 + 1) * 2 = 8
    assert chained_fn(input_value) == expected_output

    # test with a single function
    single_fn = preprocess.chain(add_one)
    assert single_fn(3) == 4

    # test with no functions should raise ValueError
    with pytest.raises(ValueError):
        preprocess.chain()

    # test with a non-callable argument should raise TypeError
    with pytest.raises(TypeError):
        preprocess.chain(add_one, None)


def test_normalize_imagenet() -> None:
    """Test the `normalize_imagenet` function for image normalization."""
    # create a dummy image with known values
    dummy_image = tf.constant(
        np.array(
            [[[0, 128, 255], [64, 192, 32]], [[255, 0, 128], [32, 64, 192]]]
        ),
        dtype=tf.uint8,
    )
    example = {"image": dummy_image}

    # normalize the image
    normalized_example = preprocess.normalize_imagenet(example)

    # expected mean and std for ImageNet
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    # manually compute expected normalized values
    expected_normalized_image = (
        tf.cast(dummy_image, tf.float32) / 255.0  # type: ignore[arg-type]
        - imagenet_mean
    ) / imagenet_std

    # check if the normalized image matches the expected values
    tf.debugging.assert_near(
        normalized_example["image"],
        expected_normalized_image,
        atol=1e-5,
    )


def test_normalize_openai() -> None:
    """Test the `normalize_openai` function for image normalization."""
    # create a dummy image with known values
    dummy_image = tf.constant(
        np.array(
            [[[0, 128, 255], [64, 192, 32]], [[255, 0, 128], [32, 64, 192]]]
        ),
        dtype=tf.uint8,
    )
    example = {"image": dummy_image}

    # normalize the image
    normalized_example = preprocess.normalize_openai(example)

    # expected mean and std for OpenAI
    openai_mean = np.array([0.48145466, 0.4578275, 0.40821073])
    openai_std = np.array([0.26862954, 0.26130258, 0.27577711])

    # manually compute expected normalized values
    expected_normalized_image = (
        tf.cast(dummy_image, tf.float32) / 255.0  # type: ignore[arg-type]
        - openai_mean
    ) / openai_std

    # check if the normalized image matches the expected values
    tf.debugging.assert_near(
        normalized_example["image"],
        expected_normalized_image,
        atol=1e-5,
    )


def test_resize() -> None:
    """Test the `resize` function for image resizing."""
    # create a dummy image with known values
    dummy_image = tf.constant(
        np.array(
            [[[0, 128, 255], [64, 192, 32]], [[255, 0, 128], [32, 64, 192]]]
        ),
        dtype=tf.uint8,
    )
    example = {"image": dummy_image}

    # resize the image to (4, 4)
    target_size = (4, 4)
    resized_example = preprocess.resize(example, size=target_size)

    # manually compute expected resized values using TensorFlow
    expected_resized_image = tf.image.resize(dummy_image, target_size)

    # check if the resized image matches the expected values
    tf.debugging.assert_near(
        resized_example["image"],
        expected_resized_image,
        atol=1e-5,
    )


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
