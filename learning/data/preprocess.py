import typing

import tensorflow as tf

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]


def chain(*args: typing.Any) -> typing.Callable:
    """Combine multiple preprocessing functions into a sequence.

    Args:
        *args: The preprocessing functions to combine.

    Returns:
        Callable: A sequence of preprocessing functions.

    Raises:
        ValueError: If no processing functions are provided.
        TypeError: If any of the provided arguments is not callable.
    """
    # ensure at least one function is provided
    if not args:
        raise ValueError("At least one processing function must be provided.")

    # ensure all arguments are callable
    for i, fn in enumerate(args):
        if not callable(fn):
            raise TypeError(f"Argument {i} is not a callable: {type(fn)}")

    def _sequential(inputs: typing.Any) -> typing.Any:
        for fn in args:
            inputs = fn(inputs)
        return inputs

    return _sequential


def filter_keys(
    example: typing.Dict[str, typing.Any],
    keys: typing.Sequence[str],
) -> typing.Dict[str, typing.Any]:
    """Filters a dictionary to only include specified keys.

    Args:
        example (Dict[str, Any]): A dictionary to filter.
        keys (Sequence[str]): A sequence of keys to retain in the dictionary.

    Returns:
        Dict[str, Any]: A filtered dictionary containing only specified keys.
    """
    return {k: v for k, v in example.items() if k in keys}


def normalize(
    example: typing.Dict[str, typing.Any],
    mean: typing.Union[float, typing.Sequence[float]],
    std: typing.Union[float, typing.Sequence[float]],
) -> typing.Dict[str, typing.Any]:
    """Normalizes an image array to the range [0, 1].

    Args:
        example (Dict[str, Any]): A dictionary containing an `image` key
            with a TensorFlow tensor.
        mean (Union[float, Sequence[float]]): A float or sequence of floats
            representing the channel means,
        std (Union[float, Sequence[float]]): A float or sequence of floats
            representing the channel standard deviations.

    Returns:
        Dict[str, Any]: A dictionary with the normalized image.
    """
    assert "image" in example, "Input dictionary must contain an 'image' key."
    image = example["image"]
    image = tf.truediv(tf.cast(image, dtype=tf.float32), 255.0)
    _mean = tf.broadcast_to(
        tf.constant(mean, dtype=tf.float32),
        shape=tf.shape(image),
    )
    _std = tf.broadcast_to(
        tf.constant(std, dtype=tf.float32),
        shape=tf.shape(image),
    )
    new_image = tf.truediv(image - _mean, _std)  # type: ignore[arg-type]
    example["image"] = new_image

    return example


def normalize_imagenet(
    example: typing.Dict[str, typing.Any],
) -> typing.Dict[str, typing.Any]:
    """Normalizes an image array using ImageNet statistics.

    Args:
        example (Dict[str, Any]): A dictionary containing an `image` key
            with a TensorFlow tensor.

    Returns:
        Dict[str, Any]: A dictionary with the normalized image.
    """
    return normalize(
        example,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )


def normalize_openai(
    example: typing.Dict[str, typing.Any],
) -> typing.Dict[str, typing.Any]:
    """Normalizes an image array using OpenAI CLIP statistics.

    Args:
        example (Dict[str, Any]): A dictionary containing an `image` key
            with a TensorFlow tensor.

    Returns:
        Dict[str, Any]: A dictionary with the normalized image.
    """
    return normalize(
        example,
        mean=OPENAI_MEAN,
        std=OPENAI_STD,
    )


def resize(
    example: typing.Dict[str, typing.Any],
    size: typing.Sequence[int],
) -> typing.Dict[str, typing.Any]:
    """Resizes an image to the specified size.

    Args:
        example (Dict[str, Any]): A dictionary containing an `image` key
            with a TensorFlow tensor.
        size (Sequence[int]): A sequence of integers specifying the
            new shape of the image.

    Returns:
        Dict[str, Any]: A dictionary with the resized image.
    """
    assert "image" in example, "Input dictionary must contain an 'image' key."
    image = example["image"]
    new_image = tf.image.resize(image, size)
    example["image"] = new_image
    return example
