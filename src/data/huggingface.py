import abc
import functools
import os
import typing

import datasets
import jax
from jax import numpy as jnp
from jax import random
import jaxtyping
from PIL import Image
import tensorflow as tf
import typing_extensions

from src.core import datamodule

# Type aliases
PyTree = jaxtyping.PyTree


class HuggingFaceDataModule(datamodule.DataModule):
    r"""A generic datamodule for HuggingFace datasets.

    To implement a new dataset, inherit from this class and implement the
    abstract methods and properties:

        - `hf_dataset`: the HuggingFace dataset object.
        - `feature_key`: the key in the dataset features to use as input.
        - `target_key`: the key in the dataset features to use as target.
        - `output_signature`: a (nested) structure of `tf.TensorSpec` objects.
        - `_create_dataset`: method to create a `tf.data.Dataset` from the
            HuggingFace dataset object.

    Attributes:
        path (str): The path to the HuggingFace dataset.
        revision (str): The revision of the dataset for version control.

    Args:
        batch_size (int): The batch size for data loading.
        deterministic (bool): Whether enforce deterministic loading behavior.
        drop_remainder (bool): Whether to drop the last incomplete batch.
        num_workers (int): Number of shards for distributed loading.
        seed (int): Random seed for shuffling.
        transform (Optional[Callable], optional): An optional function to
            transform the input features. Defaults to `None`.
        target_transform (Optional[Callable], optional): An optional function
            to transform the target features. Defaults to `None`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool,
        drop_remainder: bool,
        num_workers: int,
        seed: int,
        shuffle_buffer_size: int,
        transform: typing.Optional[typing.Callable] = None,
        target_transform: typing.Optional[typing.Callable] = None,
    ) -> None:
        self._batch_size = batch_size
        self._deterministic = deterministic
        self._drop_remainder = drop_remainder
        self._num_workers = num_workers
        self._seed = seed
        self._shuffle_buffer_size = shuffle_buffer_size
        self._rng = random.fold_in(
            random.PRNGKey(self._seed),
            jax.process_index(),
        )
        self._transform = transform
        self._target_transform = target_transform

    # =========================================
    # Interface
    @property
    @abc.abstractmethod
    def hf_dataset(self) -> datasets.DatasetDict:
        r"""datasets.DatasetDict: The HuggingFace dataset object."""
        ...

    @property
    @abc.abstractmethod
    def feature_key(self) -> str:
        r"""str: The key in the dataset features to use as input."""
        ...

    @property
    @abc.abstractmethod
    def target_key(self) -> typing.Optional[str]:
        r"""Optional[str]: The key in the dataset features to use as target."""
        ...

    @property
    @abc.abstractmethod
    def output_signature(self) -> typing.Any:
        r"""Any: A (nested) structure of `tf.TensorSpec` objects."""
        ...

    @abc.abstractmethod
    def _create_dataset(
        self,
        *,
        split: str,
        shuffle_seed: typing.Optional[int] = None,
    ) -> tf.data.Dataset:
        r"""Create an `tf.data.Dataset` from the HuggingFace dataset object.

        `Pillow` images. Override this method for custom dataset processing.

        Args:
            split (str): The dataset split to create.
            shuffle_seed (Optional[int], optional): Seed for shuffling.
                If `None`, no shuffling is applied.

        Returns:
            The created `tf.data.Dataset` instance.
        """
        pass

    # =========================================
    @property
    def batch_size(self) -> int:
        r"""int: The batch size for data loading."""
        return self._batch_size

    @property
    def deterministic(self) -> bool:
        r"""bool: Whether the dataloaders are deterministic."""
        return self._deterministic

    @property
    def drop_remainder(self) -> bool:
        r"""bool: Whether to drop the last incomplete batch."""
        return self._drop_remainder

    @property
    def num_workers(self) -> int:
        r"""int: Number of shards for distributed loading."""
        return self._num_workers

    @property
    def num_train_examples(self) -> int:
        r"""int: Number of training examples."""
        return len(self.hf_dataset["train"])  # type: ignore

    @property
    def num_val_examples(self) -> int:
        r"""int: Number of validation examples."""
        # NOTE: using test set as validation set by default
        return len(self.hf_dataset["validation"])  # type: ignore

    @property
    def num_test_examples(self) -> int:
        r"""int: Number of test examples."""
        return len(self.hf_dataset["test"])  # type: ignore

    @property
    def seed(self) -> int:
        r"""int: Random seed for shuffling."""
        return self._seed

    @property
    def shuffle_buffer_size(self) -> int:
        r"""int: Buffer size for shuffling the dataset."""
        return self._shuffle_buffer_size

    @property
    def splits(self) -> typing.List[str]:
        r"""List[str]: Available dataset splits."""
        return list(self.hf_dataset.keys())  # type: ignore

    @property
    def transform(self) -> typing.Optional[typing.Callable]:
        r"""Optional[Callable]: Transformation for the input features."""
        return self._transform

    @property
    def target_transform(self) -> typing.Optional[typing.Callable]:
        r"""Optional[Callable]: Transformation for the target features."""
        return self._target_transform

    def train_dataloader(self) -> typing.Generator[PyTree, None, None]:
        r"""Returns an iterable over the training dataset."""
        self._rng, shuffle_rng = random.split(self._rng, num=2)
        ds = self._create_dataset(
            split="train",
            shuffle_seed=int(shuffle_rng[0]),  # type: ignore
        )
        for data in ds.as_numpy_iterator():
            yield jax.tree_util.tree_map(lambda x: jnp.asarray(x), data)

    def eval_dataloader(self) -> typing.Generator[PyTree, None, None]:
        r"""Returns an iterable over the validation dataset."""
        ds = self._create_dataset(split="validation")
        for data in ds.as_numpy_iterator():
            yield jax.tree_util.tree_map(lambda x: jnp.asarray(x), data)

    def test_dataloader(self) -> typing.Generator[PyTree, None, None]:
        r"""Returns an iterable over the test dataset."""
        ds = self._create_dataset(split="test")
        for data in ds.as_numpy_iterator():
            yield jax.tree_util.tree_map(lambda x: jnp.asarray(x), data)


class HuggingFaceImageDataModule(HuggingFaceDataModule):
    r"""Data module for HuggingFace image datasets.

    Attributes:
        path (str): The path to the HuggingFace dataset.
        revision (str): The revision of the dataset for version control.

    Args:
        batch_size (int): The batch size for data loading.
        deterministic (bool): Whether the dataloaders are deterministic.
        drop_remainder (bool): Whether to drop the last incomplete batch.
        num_workers (int): Number of shards for distributed loading.
        resize (int): The size to resize images to (square).
        resample (int): Resampling filter to use for resizing images.
        transform (Optional[Callable], optional): An optional function to
            transform the input images. Defaults to `None`.
        seed (int, optional): Random seed for shuffling. Defaults to `42`.
        streaming (bool, optional): Whether to stream the dataset using the
            `datasets` library. Defaults to `False`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool,
        drop_remainder: bool,
        num_workers: int,
        resize: int,
        resample: int,
        shuffle_buffer_size: int,
        seed: int,
        transform: typing.Optional[typing.Callable] = None,
        target_transform: typing.Optional[typing.Callable] = None,
    ) -> None:
        r"""Instantiates a `HuggingFaceImageDataModule` object."""
        self._resize = resize
        self._resample = resample

        super().__init__(
            batch_size=batch_size,
            deterministic=deterministic,
            drop_remainder=drop_remainder,
            num_workers=num_workers,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            transform=transform,
            target_transform=target_transform,
        )

    def _create_dataset(
        self,
        *,
        split: str,
        shuffle_seed: typing.Optional[int] = None,
    ) -> tf.data.Dataset:
        r"""Create an `tf.data.Dataset` from the HuggingFace dataset object.

        The default method is suitable for processing image datasets with
        `Pillow` images. Override this method for custom dataset processing.

        Args:
            split (str): The dataset split to create.
            shuffle_seed (Optional[int], optional): Seed for shuffling.
                If `None`, no shuffling is applied.

        Returns:
            The created `tf.data.Dataset` instance.
        """
        _hf_dataset = self.hf_dataset[split]

        def __hf_generator() -> typing.Generator[typing.Any, None, None]:
            r"""Default iterator over HuggingFace dataset."""
            for example in _hf_dataset:
                image = example[self.feature_key]  # type: ignore
                target = (
                    example[self.target_key]  # type: ignore
                    if self.target_key
                    else None
                )
                if not isinstance(image, Image.Image):
                    raise ValueError(
                        "Default iterator expects the image to be a "
                        f"`PIL.Image.Image` object, but got {type(image)}."
                    )
                image = image.convert("RGB")

                # resize the image
                width, height = image.size
                scale = self._resize / min(width, height)
                new_width, new_height = int(width * scale), int(height * scale)
                image = image.resize(
                    size=(new_width, new_height),
                    resample=self._resample,
                )

                # center crop
                left = (new_width - self._resize) / 2
                top = (new_height - self._resize) / 2
                right = (new_width + self._resize) / 2
                bottom = (new_height + self._resize) / 2
                image = image.crop((left, top, right, bottom))

                yield image, target

        ds = tf.data.Dataset.from_generator(
            __hf_generator,
            output_signature=self.output_signature,
        )

        def __make_shard_dataset(
            shard_index: int,
            num_workers: int,
            dataset: tf.data.Dataset,
            local_seed: typing.Optional[int] = None,
        ) -> tf.data.Dataset:
            r"""Shards the input TensorFlow dataset for parallel loading."""
            local_ds = dataset.shard(num_shards=num_workers, index=shard_index)
            if local_seed is not None:
                local_ds = local_ds.shuffle(
                    buffer_size=self.shuffle_buffer_size,
                    seed=int(local_seed),  # type: ignore
                )
            if self.transform is not None:
                local_ds = local_ds.map(
                    map_func=self.transform,
                    deterministic=self.deterministic,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
            local_ds = local_ds.batch(
                batch_size=self.batch_size,
                deterministic=self.deterministic,
                drop_remainder=self.drop_remainder,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

            return local_ds

        if shuffle_seed is not None:
            local_seed = random.fold_in(
                random.PRNGKey(shuffle_seed),
                jax.process_index(),
            )[0]
            local_seed = int(local_seed)  # type: ignore
        else:
            local_seed = None

        indices = tf.data.Dataset.range(self.num_workers)
        out = indices.interleave(
            map_func=functools.partial(
                __make_shard_dataset,
                num_workers=self.num_workers,
                dataset=ds,
                local_seed=local_seed,
            ),
            deterministic=self.deterministic,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        return out.prefetch(buffer_size=tf.data.AUTOTUNE)


# ==============================================================================
# Common datasets
class ImageNet1KDataModule(HuggingFaceImageDataModule):
    r"""ILSVRC2012 image dataset subset with :math:`1,000` classes.

    The ILSVRC 2012, commonly known as 'ImageNet', is a large-scale image
    classification dataset organized according to the `WordNet` hierarchy. Each
    meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". This dataset is most commonly used **subset** of the larger ImageNet dataset, spanning over
    :math:`1,000` object categories and containing over :math:`1,281,167`
    training, :math:`50,000` validation, and :math:`100,000` testing images.

    Args:
        batch_size (int): The batch size for data loading.
        deterministic (bool, optional): Whether the dataloaders are
            deterministic. Defaults to `True`.
        drop_remainder (bool, optional): Whether to drop the last incomplete
            batch. Defaults to `True`.
        num_workers (int, optional): Number of shards for distributed loading.
            Defaults to `4`.
        resize (int, optional): The size to resize the shortest edge of the
            image to before cropping. Defaults to `224`.
        resample (int, optional): Resampling filter to use when resizing
            images. Defaults to `3` (PIL.Image.BICUBIC).
        seed (int, optional): Random seed for shuffling. Defaults to `42`.
        shuffle_buffer_size (int, optional): Buffer size for random shuffling.
            Defaults to `10_000`.
        streaming (bool, optional): Whether to stream the dataset using the
            `datasets` library. Defaults to `False`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool = True,
        drop_remainder: bool = True,
        num_workers: int = 4,
        resize: int = 224,
        resample: int = 3,
        seed: int = 42,
        shuffle_buffer_size: int = 10_000,
        streaming: bool = False,
        transform: typing.Optional[typing.Callable] = None,
        target_transform: typing.Optional[typing.Callable] = None,
    ) -> None:
        self._hf_dataset = datasets.load_dataset(
            path="ILSVRC/imagenet-1k",
            token=os.getenv("HF_TOKEN", None),
            revision="49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
            streaming=streaming,
        )
        super().__init__(
            batch_size=batch_size,
            deterministic=deterministic,
            drop_remainder=drop_remainder,
            num_workers=num_workers,
            resize=resize,
            resample=resample,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
            transform=transform,
            target_transform=target_transform,
        )

    @property
    def hf_dataset(self) -> datasets.DatasetDict:
        r"""datasets.DatasetDict: The HuggingFace dataset object."""
        return self._hf_dataset  # type: ignore

    @property
    def feature_key(self) -> str:
        r"""str: The key in the dataset features to use as input."""
        return "image"

    @property
    def target_key(self) -> str:
        r"""str: The key in the dataset features to use as target."""
        return "label"

    @property
    def output_signature(self) -> typing.Tuple[tf.TensorSpec, tf.TensorSpec]:
        r"""Tuple[tf.TensorSpec, tf.TensorSpec]: Tensor specifications."""
        return (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.uint8),  # type: ignore
            tf.TensorSpec(shape=(), dtype=tf.int64),  # type: ignore
        )


class MNISTDataModule(HuggingFaceImageDataModule):
    r"""MNIST Handwritten Digit Dataset.

    The MNIST dataset is a collection of :math:`70,000` handwritten digit images
    from :math:`0` to :math:`9`. The dataset is split into :math:`60,000`
    training examples and :math:`10,000` test examples. The dataset is available
    at `http://yann.lecun.com/exdb/mnist/`.

    Args:
        batch_size (int): The batch size for data loading.
        deterministic (bool, optional): Whether the dataloaders are
            deterministic. Defaults to `True`.
        drop_remainder (bool, optional): Whether to drop the last incomplete
            batch. Defaults to `True`.
        num_workers (int, optional): Number of shards for distributed loading.
            Defaults to `4`.
        resize (int, optional): The size to resize the shortest edge of the
            image to before cropping. Defaults to `224`.
        resample (int, optional): Resampling filter to use when resizing
            images. Defaults to `3` (PIL.Image.BICUBIC).
        seed (int, optional): Random seed for shuffling. Defaults to `42`.
        shuffle_buffer_size (int, optional): Buffer size for random shuffling.
            Defaults to `10_000`.
        streaming (bool, optional): Whether to stream the dataset using the
            `datasets` library. Defaults to `False`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool = True,
        drop_remainder: bool = True,
        num_workers: int = 4,
        resize: int = 224,
        resample: int = 3,
        seed: int = 42,
        shuffle_buffer_size: int = 10_000,
        streaming: bool = False,
        transform: typing.Optional[typing.Callable] = None,
        target_transform: typing.Optional[typing.Callable] = None,
    ) -> None:
        self._hf_dataset = datasets.load_dataset(
            path="ylecun/mnist",
            token=os.getenv("HF_TOKEN", None),
            revision="77f3279092a1c1579b2250db8eafed0ad422088c",
            streaming=streaming,
        )
        super().__init__(
            batch_size=batch_size,
            deterministic=deterministic,
            drop_remainder=drop_remainder,
            num_workers=num_workers,
            resize=resize,
            resample=resample,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
            transform=transform,
            target_transform=target_transform,
        )

    @property
    def hf_dataset(self) -> datasets.DatasetDict:
        r"""datasets.DatasetDict: The HuggingFace dataset object."""
        return self._hf_dataset  # type: ignore

    @property
    def feature_key(self) -> str:
        r"""str: The key in the dataset features to use as input."""
        return "image"

    @property
    def target_key(self) -> str:
        r"""str: The key in the dataset features to use as target."""
        return "label"

    @property
    def output_signature(self) -> typing.Tuple[tf.TensorSpec, tf.TensorSpec]:
        r"""Tuple[tf.TensorSpec, tf.TensorSpec]: Tensor specifications."""
        return (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.uint8),  # type: ignore
            tf.TensorSpec(shape=(), dtype=tf.int64),  # type: ignore
        )

    @property
    @typing_extensions.override
    def num_val_examples(self) -> int:
        r"""int: Number of validation examples."""
        # NOTE: using test set as validation set by default
        return len(self.hf_dataset["test"])  # type: ignore

    @typing_extensions.override
    def eval_dataloader(self) -> typing.Generator[PyTree, None, None]:
        r"""Returns an iterable over the validation dataset."""
        ds = self._create_dataset(split="test")
        for data in ds.as_numpy_iterator():
            yield jax.tree_util.tree_map(lambda x: jnp.asarray(x), data)


__all__ = [
    "HuggingFaceDataModule",
    "HuggingFaceImageDataModule",
    "ImageNet1KDataModule",
    "MNISTDataModule",
]
