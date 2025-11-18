import abc
import typing

# Type Aliases
EVAL_DATALODER = typing.Any
TRAIN_DATALODER = typing.Any


class DataModule(abc.ABC):
    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        r"""int: The batch size used for training and evaluation."""
        ...

    @property
    @abc.abstractmethod
    def deterministic(self) -> bool:
        r"""bool: Whether to enforce deterministic data loading."""
        ...

    @property
    @abc.abstractmethod
    def drop_remainder(self) -> bool:
        r"""bool: Whether to drop the last incomplete batch."""
        ...

    @property
    @abc.abstractmethod
    def num_workers(self) -> int:
        r"""int: The number of parallel workers for data loading."""
        ...

    @abc.abstractmethod
    def train_dataloader(self) -> TRAIN_DATALODER:
        r"""Returns an iterable over the training dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def eval_dataloader(self) -> EVAL_DATALODER:
        r"""Returns an iterable over the evaluation dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_dataloader(self) -> EVAL_DATALODER:
        r"""Returns an iterable over the test dataset."""
        raise NotImplementedError
