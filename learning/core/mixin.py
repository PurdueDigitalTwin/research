import abc
import typing

from flax import linen as nn
from flax import struct
from flax.core import frozen_dict
from jax import random
import jaxtyping

# Type Aliases
PyTree = jaxtyping.PyTree
EVAL_DATALOADER = typing.Any
TRAIN_DATALOADER = typing.Any


class DataMixin:
    """Trait mixin for data loading and preprocessing."""

    def prepare_data(self) -> None:
        """Downloads and prepares the dataset."""
        pass

    @abc.abstractmethod
    def train_dataloader(self) -> TRAIN_DATALOADER:
        """Returns the training dataloader."""
        raise NotImplementedError

    @abc.abstractmethod
    def val_dataloader(self) -> EVAL_DATALOADER:
        """Returns the validation dataloader."""
        raise NotImplementedError

    @abc.abstractmethod
    def test_dataloader(self) -> EVAL_DATALOADER:
        """Returns the test dataloader."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input_features(self) -> PyTree:
        """PyTree: A dictionary mapping feature names to example arrays."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def target_features(self) -> PyTree:
        """PyTree: A dictionary mapping feature names to example arrays."""
        raise NotImplementedError


class ModelMixin:
    """Trait for model definition and initialization."""

    module_class: typing.Type[nn.Module]
    """Type[nn.Module]: The class of the model module."""
    _module: nn.Module = struct.field(pytree_node=True)
    """nn.Module: The model module instance."""

    def __call__(
        self,
        *,
        params: frozen_dict.FrozenDict,
        deterministic: bool = True,
        **kwargs,
    ) -> typing.Any:
        """Calls the model by forward passing the network module.

        Args:
            params (FrozenDict): The model parameters.
            deterministic (bool): Whether to run the model in deterministic
                mode (e.g., disable dropout). Default is `True`.
            **kwargs: Keyword arguments consumed by the model.

        Returns:
            Any: The model output.
        """
        return self.forward(
            params=params,
            deterministic=deterministic,
            **kwargs,
        )

    @abc.abstractmethod
    def compute_loss(
        self,
        *,
        rngs: typing.Union[
            random.KeyArray,
            typing.Dict[str, random.KeyArray],
        ],
        deterministic: bool = False,
        params: frozen_dict.FrozenDict,
        **kwargs,
    ) -> PyTree:
        """Computes the loss given parameters and model inputs.

        Args:
            deterministic (bool): Whether to run the model in deterministic
                mode (e.g., disable dropout). Default is `False`.
            params (FrozenDict): The model parameters.
            **kwargs: Keyword arguments consumed by the model.

        Returns:
            PyTree: A dictionary containing the loss and other outputs.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dummy_input(self) -> PyTree:
        """Returns a dummy input for model initialization."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        *,
        rngs: typing.Union[
            random.KeyArray,
            typing.Dict[str, random.KeyArray],
        ],
        deterministic: bool = True,
        params: frozen_dict.FrozenDict,
        **kwargs,
    ) -> PyTree:
        """Forward pass the model and returns the output tree structure.

        Args:
            deterministic (bool): Whether to run the model in deterministic
                mode (e.g., disable dropout). Default is `True`.
            params (FrozenDict): The model parameters.
            **kwargs: Keyword arguments consumed by the model.

        Returns:
            PyTree: The model output.
        """
        raise NotImplementedError

    def init(
        self,
        rngs: typing.Union[random.KeyArray, typing.Dict[str, random.KeyArray]],
        print_summary: bool = True,
    ) -> frozen_dict.FrozenDict:
        """Initializes the model parameters.

        Args:
            rngs (Union[KeyArray, Dict[str, KeyArray]): The random number
                generator(s) for model initialization.

        Returns:
            FrozenDict: The initialized model parameters.
        """
        variables = self._module.init(
            rngs,
            *self.dummy_input.values(),
            deterministic=True,
        )
        if print_summary:
            # print model summary to console
            _tabulate_fn = nn.summary.tabulate(self._module, rngs=rngs)
            print(_tabulate_fn(**self.dummy_input, deterministic=True))

        return variables["params"]

    # Outer-loop functions
    def on_train_epoch_start(self) -> None:
        """Called at the beginning of the training epoch."""
        pass

    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch."""
        pass

    def on_validation_epoch_start(self) -> None:
        """Called at the beginning of the validation epoch."""
        pass

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch."""
        pass

    def on_test_epoch_start(self) -> None:
        """Called at the beginning of the test epoch."""
        pass

    def on_test_epoch_end(self) -> None:
        """Called at the end of the test epoch."""
        pass

    # Inner-loop functions
    def on_train_step_start(self) -> None:
        """Called at the beginning of the training step."""
        pass

    def on_train_step_end(self) -> None:
        """Called at the end of the training step."""
        pass

    def on_validation_step_start(self) -> None:
        """Called at the beginning of the validation step."""
        pass

    def on_validation_step_end(self) -> None:
        """Called at the end of the validation step."""
        pass

    def on_test_step_start(self) -> None:
        """Called at the beginning of the test step."""
        pass

    def on_test_step_end(self) -> None:
        """Called at the end of the test step."""
        pass
