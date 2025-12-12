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
import tensorflow as tf
import typing_extensions

from src.core import datamodule

# Type aliases
PyTree = jaxtyping.PyTree


# ==============================================================================
# Helper Functions
def _align_keys(key: str) -> str:
    r"""Aligns common feature keys to standard names."""

    key_mappings = {
        "label_ids": "labels",
        "label_id": "labels",
        "target": "labels",
        "targets": "labels",
        "image": "image",
        "img": "image",
        "images": "image",
    }

    return key_mappings.get(key, key)


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
        _align_keys(key): value
        for key, value in data.items()
        if key in columns or key in ("label", "label_ids", "labels")
    }

    # enforce data types
    out = []
    for col, cast_dtype in columns_dtypes.items():
        arr = np.array(data[_align_keys(col)]).astype(cast_dtype)
        if _align_keys(col) == "image":
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
                arr = np.tile(arr, (1, 1, 3))  # convert grayscale to RGB
            if arr.shape[-1] == 4:
                arr = arr[..., :3]  # remove alpha channel
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
    def feature_keys(self) -> typing.List[str]:
        r"""List[str]: The keys in the dataset features to use as input."""
        ...

    @property
    @abc.abstractmethod
    def feature_types(self) -> typing.Dict[str, typing.Any]:
        r"""Dict[str, Any]: Mapping of feature keys to their types."""
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
        shuffle_buffer_size: int,
        transform: typing.Optional[typing.Callable] = None,
        use_cache: bool = False,
        rng: typing.Any = jax.random.PRNGKey(42),
    ) -> None:
        if use_cache:
            cache_dir = os.path.join(
                tempfile.gettempdir(),
                "huggingface",
                "datasets",
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
        self._train_dataset = self.create_dataset(
            batch_size=self.batch_size,
            dataset=self.hf_dataset["train"],
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
            dataset=self.hf_dataset["test"],
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
                dataset=self.hf_dataset["validation"],
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
                dataset=self.hf_dataset["val"],
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

    def create_dataset(
        self,
        *,
        dataset: datasets.Dataset,
        batch_size: int,
        deterministic: bool,
        drop_remainder: bool,
        shuffle_buffer_size: int,
        shuffle_seed: typing.Optional[int] = None,
        transform: typing.Optional[typing.Callable] = None,
        cache_dir: typing.Optional[str] = None,
    ) -> tf.data.Dataset:
        r"""Create `tf.data.Dataset` from the HuggingFace dataset.

        .. note::
            **Breaking change:** The `create_dataset` method signature has
            changed. The `dataset` parameter type is now `datasets.Dataset`
            (from HuggingFace), not `tf.data.Dataset`, and its position in the
            parameter order has changed (it is now a keyword-only argument).
            Update any overrides or calls accordingly.


        Args:
            dataset (datasets.Dataset): The HuggingFace dataset.
            batch_size (int): The batch size for data loading.
            deterministic (bool): Whether to enforce deterministic loading.
            drop_remainder (bool): Whether to drop the last incomplete batch.

            shuffle_buffer_size (int): Buffer size for random shuffling.
            shuffle_seed (Optional[int], optional): Seed for shuffling.
                If `None`, no shuffling is applied.
            transform (Optional[Callable], optional): An optional function to
                transform the features. Default is `None`.
            cache_dir (Optional[str], optional): Directory to cache the dataset.

        Returns:
            The created `tf.data.Dataset` instance.
        """

        # step 1: map fetch function to get data from huggingface dataset
        get_fn = functools.partial(
            _hf_dataset_get,
            dataset=dataset,
            columns=self.feature_keys,
            columns_dtypes=self.feature_types,
        )
        tout = [tf.dtypes.as_dtype(t) for t in self.feature_types.values()]

        @tf.function(
            input_signature=(tf.TensorSpec(None, tf.int64),)  # type: ignore
        )
        def fetch_fn(index: tf.Tensor) -> typing.Dict[str, tf.Tensor]:
            output = tf.py_function(
                get_fn,
                inp=[index],
                Tout=tout,
            )
            return {
                _align_keys(key): output[i]  # type: ignore
                for i, key in enumerate(self.feature_keys)
            }

        ds = tf.data.Dataset.range(len(dataset))
        ds = ds.map(
            map_func=fetch_fn,
            deterministic=deterministic,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # step 2: map transformation function to preprocess images
        if isinstance(transform, typing.Callable):
            ds = ds.map(
                map_func=transform,
                deterministic=deterministic,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if shuffle_seed is not None:
            ds = ds.shuffle(
                buffer_size=shuffle_buffer_size,
                seed=shuffle_seed,
                reshuffle_each_iteration=True,
            )

        if cache_dir is not None:
            ds = ds.cache(filename=cache_dir)

        ds = ds.batch(
            batch_size=batch_size,
            deterministic=deterministic,
            drop_remainder=drop_remainder,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

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
        shuffle_buffer_size (int, optional): Buffer size for random shuffling.
            Defaults to `10_000`.
        streaming (bool, optional): Whether to stream the dataset using the
            `datasets` library. Defaults to `False`.
        use_cache (bool, optional): Whether to use cached dataset.
            Default is `False`.
        rng (jax.Array, optional): Random key for shuffling.
            Default is `random.PRNGKey(42)`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool = True,
        drop_remainder: bool = True,
        num_workers: int = 4,
        shuffle_buffer_size: int = 10_000,
        streaming: bool = False,
        transform: typing.Optional[typing.Callable] = None,
        use_cache: bool = False,
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
            shuffle_buffer_size=shuffle_buffer_size,
            transform=transform,
            use_cache=use_cache,
            rng=rng,
        )

    @property
    def hf_dataset(self) -> datasets.DatasetDict:
        r"""datasets.DatasetDict: The HuggingFace dataset object."""
        return self._hf_dataset  # type: ignore

    @property
    def feature_keys(self) -> typing.List[str]:
        return ["img", "label"]

    @property
    def feature_types(self) -> typing.Dict[str, typing.Any]:
        return {"img": np.uint8, "label": np.int32}

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
        seed (int, optional): Random seed for shuffling. Defaults to `42`.
        shuffle_buffer_size (int, optional): Buffer size for random shuffling.
            Defaults to `10_000`.
        streaming (bool, optional): Whether to stream the dataset using the
            `datasets` library. Defaults to `False`.
        use_cache (bool, optional): Whether to use cached dataset.
            Default is `False`.
        rng (jax.Array, optional): Random key for shuffling.
            Defaults to `random.PRNGKey(42)`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool = True,
        drop_remainder: bool = True,
        num_workers: int = 4,
        shuffle_buffer_size: int = 10_000,
        streaming: bool = False,
        transform: typing.Optional[typing.Callable] = None,
        use_cache: bool = False,
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
            shuffle_buffer_size=shuffle_buffer_size,
            transform=transform,
            use_cache=use_cache,
            rng=rng,
        )

    @property
    def hf_dataset(self) -> datasets.DatasetDict:
        r"""datasets.DatasetDict: The HuggingFace dataset object."""
        return self._hf_dataset  # type: ignore

    @property
    def feature_keys(self) -> typing.List[str]:
        return ["img", "fine_label"]

    @property
    def feature_types(self) -> typing.Dict[str, typing.Any]:
        return {"img": np.uint8, "fine_label": np.int32}

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
        shuffle_buffer_size (int, optional): Buffer size for random shuffling.
            Defaults to `10_000`.
        streaming (bool, optional): Whether to stream the dataset using the
            `datasets` library. Defaults to `False`.
        use_cache (bool, optional): Whether to use cached dataset.
            Default is `False`.
        rng (jax.Array, optional): Random key for shuffling.
            Default is `random.PRNGKey(42)`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool = True,
        drop_remainder: bool = True,
        num_workers: int = 4,
        shuffle_buffer_size: int = 10_000,
        streaming: bool = False,
        transform: typing.Optional[typing.Callable] = None,
        use_cache: bool = False,
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
            shuffle_buffer_size=shuffle_buffer_size,
            transform=transform,
            use_cache=use_cache,
            rng=rng,
        )

    @property
    def hf_dataset(self) -> datasets.DatasetDict:
        r"""datasets.DatasetDict: The HuggingFace dataset object."""
        return self._hf_dataset  # type: ignore

    @property
    def feature_keys(self) -> typing.List[str]:
        return ["image", "label"]

    @property
    def feature_types(self) -> typing.Dict[str, typing.Any]:
        return {"image": np.uint8, "label": np.int32}


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
        shuffle_buffer_size (int, optional): Buffer size for random shuffling.
            Defaults to `10_000`.
        streaming (bool, optional): Whether to stream the dataset using the
            `datasets` library. Defaults to `False`.
        use_cache (bool, optional): Whether to use cached dataset.
            Default is `False`.
        rng (jax.Array, optional): Random key for shuffling.
            Default is `random.PRNGKey(42)`.
    """

    def __init__(
        self,
        batch_size: int,
        deterministic: bool = True,
        drop_remainder: bool = True,
        num_workers: int = 4,
        shuffle_buffer_size: int = 10_000,
        streaming: bool = False,
        transform: typing.Optional[typing.Callable] = None,
        use_cache: bool = False,
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
            shuffle_buffer_size=shuffle_buffer_size,
            transform=transform,
            use_cache=use_cache,
            rng=rng,
        )

    @property
    def hf_dataset(self) -> datasets.DatasetDict:
        r"""datasets.DatasetDict: The HuggingFace dataset object."""
        return self._hf_dataset  # type: ignore

    @property
    def feature_keys(self) -> typing.List[str]:
        return ["image", "label"]

    @property
    def feature_types(self) -> typing.Dict[str, typing.Any]:
        return {"image": np.uint8, "label": np.int32}

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
