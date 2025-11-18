import abc
import typing

import chex
from flax import struct
import jaxtyping

from src.core import train_state as _train_state


@chex.dataclass
class StepOutputs:
    """A base container for outputs from a single step.

    Attributes:
        scalars (Optional[Dict[str, Any]]): A dictionary of scalar metrics.
        images (Optional[Dict[str, Any]]): A dictionary of image outputs.
    """

    scalars: typing.Optional[typing.Dict[str, typing.Any]] = None
    images: typing.Optional[typing.Dict[str, typing.Any]] = None


class Model(abc.ABC):
    """Interface for models."""

    @property
    @abc.abstractmethod
    def network(self) -> typing.Callable:
        r"""Callable: The neural network module."""
        ...

    @abc.abstractmethod
    def init(
        self,
        *,
        batch: typing.Any,
        rngs: typing.Union[typing.Any, typing.Dict[str, typing.Any]],
        **kwargs,
    ) -> jaxtyping.PyTree:
        r"""Initializes the model parameters.

        Args:
            batch (Any): An example batch of data for initialization.
            rngs (Union[Any, Dict[str, Any]]): Random generators.
            **kwargs: Additional keyword arguments.

        Returns:
            A `PyTree` the initialized model parameters.
        """
        pass

    @abc.abstractmethod
    def training_step(
        self,
        *,
        state: _train_state.TrainState,
        batch: typing.Any,
        rngs: typing.Union[typing.Any, typing.Dict[str, typing.Any]],
        **kwargs,
    ) -> typing.Tuple[struct.PyTreeNode, StepOutputs]:
        r"""Performs a single training step.

        Args:
            state (TrainState): The current training state.
            batch (Any): A batch of data.
            rngs (Union[Any, Dict[str, Any]]): Random generators.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing the updated state and step outputs.
        """
        pass

    @abc.abstractmethod
    def evaluation_step(
        self,
        *,
        params: jaxtyping.PyTree,
        batch: typing.Any,
        rngs: typing.Union[typing.Any, typing.Dict[str, typing.Any]],
        **kwargs,
    ) -> StepOutputs:
        r"""Performs a single evaluation step.

        Args:
            params (PyTree): The model parameters.
            batch (Any): A batch of data.
            rngs (Union[Any, Dict[str, Any]]): Random generators.
            **kwargs: Additional keyword arguments.

        Returns:
            The step outputs containing evaluation metrics.
        """
        pass

    @abc.abstractmethod
    def predict_step(
        self,
        *,
        params: jaxtyping.PyTree,
        batch: typing.Any,
        rngs: typing.Union[typing.Any, typing.Dict[str, typing.Any]],
        **kwargs,
    ) -> typing.Any:
        r"""Performs a single prediction step during inference.

        Args:
            params (PyTree): The model parameters.
            batch (Any): A batch of data.
            rngs (Union[Any, Dict[str, Any]]): Random generators.
            **kwargs: Additional keyword arguments.

        Returns:
            The model's predictions.
        """
        pass
