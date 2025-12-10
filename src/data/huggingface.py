import abc
import functools
import os
import shutil
import tempfile
import typing

import datasets
import jax
from jax import numpy as jnp
from jax import random
import jaxtyping
from numpy import typing as npt
import numpy as np
from PIL import Image
import tensorflow as tf
import typing_extensions

from src.core import datamodule

# Type aliases
PyTree = jaxtyping.PyTree


# ==============================================================================
# Helper Functions
def _hf_dataset_get(
    index: tf.Tensor,
    dataset: datasets.Dataset,
    columns: typing.List[str],
    columns_dtypes: typing.Dict[str, typing.Any],
) -> typing.List[npt.NDArray]:
    r"""Getter function for HuggingFace datasets."""
    index = index.numpy()  # type: ignore
    if not isinstance(index, np.integer):
        raise ValueError(
            f"`_hf_dataset_get` expects an integer index, but got {index}."
        )
    data: typing.Dict[str, npt.NDArray] = dataset[index.item()]
    data = {
        key: value
        for key, value in data.items()
        if key in columns or key in ("label", "label_ids", "labels")
    }

    # enforece data types
    out = []
    for col, cast_dtype in columns_dtypes.items():
        arr = np.array(data[col]).astype(cast_dtype)
        out.append(arr)

    return out


# ==============================================================================
# Data Modules
class HuggingFaceDataModule(datamodule.DataModule):
    r"""A generic datamodule for HuggingFace datasets.

    To implement a new dataset, inherit from this class and implement the
    abstract methods and properties:

        - `hf_dataset`: the HuggingFace dataset object.
        - `feature_key`: the key in the dataset features to use as input.
        - `target_key`: the key in the dataset features to use as target.
        - `create_dataset`: method to create a `tf.data.Dataset` from the
            HuggingFace dataset object.

    Args:
        batch_size (int): The batch size for data loading.
        deterministic (bool): Whether enforce deterministic loading behavior.
        drop_remainder (bool): Whether to drop the last incomplete batch.
        num_workers (int): Number of shards for distributed loading.
        transform (Optional[Callable], optional): An optional function to
            transform the features. Default is `None`.
        shuffle_buffer_size (int): Buffer size for shuffling the dataset.
        rng (Any): Random seed for shuffling. Default is `PRNGKey(42)`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool,
        drop_remainder: bool,
        num_workers: int,
        shuffle_buffer_size: int,
        transform: typing.Optional[typing.Callable] = None,
        rng: typing.Any = jax.random.PRNGKey(42),
    ) -> None:
        self._batch_size = batch_size
        self._deterministic = deterministic
        self._drop_remainder = drop_remainder
        self._num_workers = num_workers
        self._shuffle_buffer_size = shuffle_buffer_size
        self._rng = jax.random.fold_in(rng, jax.process_index())
        self._transform = transform

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
    def train_dataset(self) -> typing.Iterable:
        r"""Iterable: The training dataset split."""
        ...

    @property
    @abc.abstractmethod
    def eval_dataset(self) -> typing.Iterable:
        r"""Iterable: The validation dataset split."""
        ...

    @property
    @abc.abstractmethod
    def test_dataset(self) -> typing.Iterable:
        r"""Iterable: The test dataset split."""
        ...

    @staticmethod
    @abc.abstractmethod
    def create_dataset(*args, **kwargs) -> tf.data.Dataset:
        r"""Create sharded `tf.data.Dataset` from the HuggingFace dataset.

        The default method is suitable for processing image datasets with
        `Pillow` images. Override this method for custom dataset processing.

        Returns:
            The created `tf.data.Dataset` instance.
        """
        ...

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
        r"""int: Number of workers for distributed loading."""
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
    def rng(self) -> typing.Any:
        r"""Any: Random seed for shuffling."""
        return self._rng

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
        shuffle_buffer_size (int): Buffer size for random shuffling.
        transform (Optional[Callable], optional): An optional function to
            transform the input images. Default is `None`.
        use_cache (bool, optional): Whether to use cached dataset.
            Default is `True`.
        rng (Any): Random seed for shuffling. Default is `PRNGKey(42)`.
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
        transform: typing.Optional[typing.Callable] = None,
        use_cache: bool = True,
        rng: typing.Any = jax.random.PRNGKey(42),
    ) -> None:
        self._resize = resize
        self._resample = resample
        if use_cache:
            cache_dir = os.path.join(
                tempfile.gettempdir(),
                "chimera",
                "huggingface",
            )
            if os.path.exists(cache_dir):
                # NOTE: clear the cache directory to avoid corrupted cache
                shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
        else:
            cache_dir = None

        super().__init__(
            batch_size=batch_size,
            deterministic=deterministic,
            drop_remainder=drop_remainder,
            num_workers=num_workers,
            shuffle_buffer_size=shuffle_buffer_size,
            transform=transform,
            rng=rng,
        )

        # prepare the dataset splits
        pre_transform = functools.partial(
            self.pre_transform,
            feature_key=self.feature_key,
            target_key=self.target_key,
            center_crop=True,
            resample=self._resample,
            resize=self._resize,
        )
        self._train_dataset = self.create_dataset(
            batch_size=self.batch_size,
            dataset=self.hf_dataset["train"]
            .map(pre_transform, batched=False, num_proc=1)
            .to_tf_dataset(batch_size=None, prefetch=False),
            deterministic=self.deterministic,
            drop_remainder=self.drop_remainder,
            shuffle_buffer_size=self.shuffle_buffer_size,
            shuffle_seed=int(self._rng[0]),
            transform=self.transform,
            cache_dir=(
                os.path.join(cache_dir, "train_" + self.__class__.__name__)
                if cache_dir is not None
                else None
            ),
        )
        self._test_dataset = self.create_dataset(
            batch_size=self.batch_size,
            dataset=self.hf_dataset["test"]
            .map(pre_transform, batched=False, num_proc=1)
            .to_tf_dataset(batch_size=None, prefetch=False),
            deterministic=self.deterministic,
            drop_remainder=self.drop_remainder,
            shuffle_buffer_size=self.shuffle_buffer_size,
            shuffle_seed=None,
            transform=self.transform,
            cache_dir=(
                os.path.join(cache_dir, "test_" + self.__class__.__name__)
                if cache_dir is not None
                else None
            ),
        )
        if "validation" in self.hf_dataset:
            self._eval_dataset = self.create_dataset(
                batch_size=self.batch_size,
                dataset=self.hf_dataset["validation"]
                .map(pre_transform, batched=False, num_proc=1)
                .to_tf_dataset(batch_size=None, prefetch=False),
                deterministic=self.deterministic,
                drop_remainder=self.drop_remainder,
                shuffle_buffer_size=self.shuffle_buffer_size,
                shuffle_seed=None,
                transform=self.transform,
                cache_dir=(
                    os.path.join(cache_dir, "val_" + self.__class__.__name__)
                    if cache_dir is not None
                    else None
                ),
            )
        elif "val" in self.hf_dataset:
            self._eval_dataset = self.create_dataset(
                batch_size=self.batch_size,
                dataset=self.hf_dataset["val"]
                .map(pre_transform, batched=False, num_proc=1)
                .to_tf_dataset(batch_size=None, prefetch=False),
                deterministic=self.deterministic,
                drop_remainder=self.drop_remainder,
                shuffle_buffer_size=self.shuffle_buffer_size,
                shuffle_seed=None,
                transform=self.transform,
                cache_dir=(
                    os.path.join(cache_dir, "val_" + self.__class__.__name__)
                    if cache_dir is not None
                    else None
                ),
            )
        else:
            # NOTE: otherwise, use test set as validation set by default
            self._eval_dataset = self._test_dataset

    @property
    def train_dataset(self) -> tf.data.Dataset:
        r"""tf.data.Dataset: The training dataset split."""
        return self._train_dataset

    @property
    def eval_dataset(self) -> tf.data.Dataset:
        r"""tf.data.Dataset: The validation dataset split."""
        return self._eval_dataset

    @property
    def test_dataset(self) -> tf.data.Dataset:
        r"""tf.data.Dataset: The test dataset split."""
        return self._test_dataset

    @staticmethod
    def create_dataset(
        *,
        batch_size: int,
        deterministic: bool,
        drop_remainder: bool,
        dataset: tf.data.Dataset,
        shuffle_buffer_size: int,
        shuffle_seed: typing.Optional[int] = None,
        transform: typing.Optional[typing.Callable] = None,
        cache_dir: typing.Optional[str] = None,
    ) -> tf.data.Dataset:
        r"""Create sharded `tf.data.Dataset` from the HuggingFace dataset.

        The default method is suitable for processing image datasets with
        `Pillow` images. Override this method for custom dataset processing.

        Args:
            batch_size (int): The batch size for data loading.
            deterministic (bool): Whether to enforce deterministic loading.
            drop_remainder (bool): Whether to drop the last incomplete batch.
            dataset (tf.data.Dataset): The converted HuggingFace dataset.
            shuffle_buffer_size (int): Buffer size for random shuffling.
            shuffle_seed (Optional[int], optional): Seed for shuffling.
                If `None`, no shuffling is applied.
            transform (Optional[Callable], optional): An optional function to
                transform the features. Default is `None`.
            cache_dir (Optional[str], optional): Directory to cache the dataset.

        Returns:
            The created `tf.data.Dataset` instance.
        """
        if isinstance(transform, typing.Callable):
            dataset = dataset.map(
                map_func=transform,
                deterministic=deterministic,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if shuffle_seed is not None:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size,
                seed=shuffle_seed,
                reshuffle_each_iteration=True,
            )

        if cache_dir is not None:
            dataset = dataset.cache(filename=cache_dir)

        dataset = dataset.batch(
            batch_size=batch_size,
            deterministic=deterministic,
            drop_remainder=drop_remainder,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    @staticmethod
    def pre_transform(
        example: typing.Dict[str, tf.Tensor],
    ) -> typing.Dict[str, typing.Any]:
        r"""Pre-transformation function to align channels of input images.

        Args:
            example (Dict[str, tf.Tensor]): A dictionary of data tensors.

        Returns:
            A dictionary with processed images and targets.
        """
        image: tf.Tensor = example["image"]

        # handle grayscale images
        image = tf.cond(  # type: ignore
            tf.equal(tf.rank(image), 2),
            lambda: tf.expand_dims(image, axis=-1),
            lambda: image,
        )

        # align the channel dimension to 3 (RGB)
        # repeat last dimension of grayscale image (1 channel) -> RGB
        # drop alpha channel (4 channels) -> RGB
        channels = tf.shape(image)[-1]  # type: ignore
        image = tf.cond(  # type: ignore
            tf.equal(channels, 1),
            lambda: tf.image.grayscale_to_rgb(image),
            lambda: tf.cond(
                tf.equal(channels, 4),
                lambda: image[..., :3],  # type: ignore
                lambda: image,
            ),
        )

        # explicitly hint the shape
        image.set_shape([None, None, 3])

        return {"image": image, **example}

    def train_dataloader(self) -> typing.Generator[PyTree, None, None]:
        r"""Generator[PyTree]: Returns an iterable over the training data."""
        for data in self.train_dataset.as_numpy_iterator():
            yield jax.tree_util.tree_map(lambda x: jnp.asarray(x), data)

    def eval_dataloader(self) -> typing.Generator[PyTree, None, None]:
        r"""Generator[PyTree]: Returns an iterable over the validation data."""
        for data in self.eval_dataset.as_numpy_iterator():
            yield jax.tree_util.tree_map(lambda x: jnp.asarray(x), data)

    def test_dataloader(self) -> typing.Generator[PyTree, None, None]:
        r"""Generator[PyTree]: Returns an iterable over the test data."""
        for data in self.test_dataset.as_numpy_iterator():
            yield jax.tree_util.tree_map(lambda x: jnp.asarray(x), data)


# ==============================================================================
# Common datasets
class CIFAR10DataModule(HuggingFaceImageDataModule):
    r"""CIFAR-10 Image Classification Dataset.

    The CIFAR-10 dataset consists of :math:`60,000` 32x32 colour images in
    :math:`10` classes, with :math:`6,000` images per class. There are
    :math:`50,000` training images and :math:`10,000` test images. The dataset
    is divided into five training batches and one test batch, each with
    :math:`10,000` images. The test batch contains exactly :math:`1000`
    randomly-selected images from each class. The training batches contain the
    remaining images in random order, but some training batches may contain
    more images from one class than another. Between them, the training batches
    contain exactly :math:`5000` images from each class.

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
        shuffle_buffer_size (int, optional): Buffer size for random shuffling.
            Defaults to `10_000`.
        streaming (bool, optional): Whether to stream the dataset using the
            `datasets` library. Defaults to `False`.
        rng (jax.Array, optional): Random key for shuffling.
            Default is `random.PRNGKey(42)`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool = True,
        drop_remainder: bool = True,
        num_workers: int = 4,
        resize: int = 224,
        resample: int = 3,
        shuffle_buffer_size: int = 10_000,
        streaming: bool = False,
        transform: typing.Optional[typing.Callable] = None,
        rng: jax.Array = random.PRNGKey(42),
    ) -> None:
        self._hf_dataset = datasets.load_dataset(
            path="uoft-cs/cifar10",
            token=os.getenv("HF_TOKEN", None),
            revision="0b2714987fa478483af9968de7c934580d0bb9a2",
            streaming=streaming,
        )
        super().__init__(
            batch_size=batch_size,
            deterministic=deterministic,
            drop_remainder=drop_remainder,
            num_workers=num_workers,
            resize=resize,
            resample=resample,
            shuffle_buffer_size=shuffle_buffer_size,
            transform=transform,
            rng=rng,
        )

    @property
    def hf_dataset(self) -> datasets.DatasetDict:
        r"""datasets.DatasetDict: The HuggingFace dataset object."""
        return self._hf_dataset  # type: ignore

    @property
    def feature_key(self) -> str:
        r"""str: The key in the dataset features to use as input."""
        return "img"

    @property
    def target_key(self) -> str:
        r"""str: The key in the dataset features to use as target."""
        return "label"

    @property
    @typing_extensions.override
    def num_val_examples(self) -> int:
        r"""int: Number of validation examples."""
        # NOTE: using test set as validation set by default
        return len(self.hf_dataset["test"])  # type: ignore


class CIFAR100DataModule(HuggingFaceImageDataModule):
    r"""CIFAR-100 Image Classification Dataset.

    The CIFAR-100 dataset consists of :math:`60,000` 32x32 colour images in
    :math:`100` classes, with :math:`600` images per class. There are
    :math:`500` training images and :math:`100` testing images per class.
    There are :math:`50,000` training images and :math:`10,000` test images.
    The :math:`100` classes are grouped into :math:`20` superclasses.
    There are two labels per image - fine label and coarse label (superclass).

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
        rng (jax.Array, optional): Random key for shuffling.
            Defaults to `random.PRNGKey(42)`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool = True,
        drop_remainder: bool = True,
        num_workers: int = 4,
        resize: int = 224,
        resample: int = 3,
        shuffle_buffer_size: int = 10_000,
        streaming: bool = False,
        transform: typing.Optional[typing.Callable] = None,
        rng: jax.Array = random.PRNGKey(42),
    ) -> None:
        self._hf_dataset = datasets.load_dataset(
            path="uoft-cs/cifar100",
            token=os.getenv("HF_TOKEN", None),
            revision="aadb3af77e9048adbea6b47c21a81e47dd092ae5",
            streaming=streaming,
        )
        super().__init__(
            batch_size=batch_size,
            deterministic=deterministic,
            drop_remainder=drop_remainder,
            num_workers=num_workers,
            resize=resize,
            resample=resample,
            shuffle_buffer_size=shuffle_buffer_size,
            transform=transform,
            rng=rng,
        )

    @property
    def hf_dataset(self) -> datasets.DatasetDict:
        r"""datasets.DatasetDict: The HuggingFace dataset object."""
        return self._hf_dataset  # type: ignore

    @property
    def feature_key(self) -> str:
        r"""str: The key in the dataset features to use as input."""
        return "img"

    @property
    def target_key(self) -> str:
        r"""str: The key in the dataset features to use as target."""
        return "fine_label"

    @property
    @typing_extensions.override
    def num_val_examples(self) -> int:
        r"""int: Number of validation examples."""
        # NOTE: using test set as validation set by default
        return len(self.hf_dataset["test"])  # type: ignore


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
        shuffle_buffer_size (int, optional): Buffer size for random shuffling.
            Defaults to `10_000`.
        streaming (bool, optional): Whether to stream the dataset using the
            `datasets` library. Defaults to `False`.
        rng (jax.Array, optional): Random key for shuffling.
            Default is `random.PRNGKey(42)`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool = True,
        drop_remainder: bool = True,
        num_workers: int = 4,
        resize: int = 224,
        resample: int = 3,
        shuffle_buffer_size: int = 10_000,
        streaming: bool = False,
        transform: typing.Optional[typing.Callable] = None,
        rng: jax.Array = random.PRNGKey(42),
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
            shuffle_buffer_size=shuffle_buffer_size,
            transform=transform,
            rng=rng,
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
        shuffle_buffer_size (int, optional): Buffer size for random shuffling.
            Defaults to `10_000`.
        streaming (bool, optional): Whether to stream the dataset using the
            `datasets` library. Defaults to `False`.
        rng (jax.Array, optional): Random key for shuffling.
            Default is `random.PRNGKey(42)`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool = True,
        drop_remainder: bool = True,
        num_workers: int = 4,
        resize: int = 224,
        resample: int = 3,
        shuffle_buffer_size: int = 10_000,
        streaming: bool = False,
        transform: typing.Optional[typing.Callable] = None,
        rng: jax.Array = random.PRNGKey(42),
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
            shuffle_buffer_size=shuffle_buffer_size,
            transform=transform,
            rng=rng,
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
    @typing_extensions.override
    def num_val_examples(self) -> int:
        r"""int: Number of validation examples."""
        # NOTE: using test set as validation set by default
        return len(self.hf_dataset["test"])  # type: ignore


__all__ = [
    "HuggingFaceDataModule",
    "HuggingFaceImageDataModule",
    "CIFAR10DataModule",
    "CIFAR100DataModule",
    "ImageNet1KDataModule",
    "MNISTDataModule",
]
