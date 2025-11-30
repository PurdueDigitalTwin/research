import abc
import typing

import chex
from flax.core import frozen_dict
import jax
import jaxtyping


@chex.dataclass
class StepOutputs:
    """A base container for outputs from a single step.

    Attributes:
        output (Optional[jax.Array]): The main output of the model.
        scalars (Optional[Dict[str, Any]]): A dictionary of scalar metrics.
        images (Optional[Dict[str, Any]]): A dictionary of image outputs.
    """

    output: typing.Optional[jax.Array] = None
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
    def compute_loss(
        self,
        *,
        rngs: typing.Any,
        deterministic: bool = False,
        params: frozen_dict.FrozenDict,
        **kwargs,
    ) -> typing.Tuple[jax.Array, StepOutputs]:
        """Computes the loss given parameters and model inputs.

        Args:
            deterministic (bool): Whether to run the model in deterministic
                mode (e.g., disable dropout). Default is `False`.
            params (FrozenDict): The model parameters.
            **kwargs: Keyword arguments consumed by the model.

        Returns:
            A dictionary containing the loss and other outputs.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        *,
        rngs: typing.Any,
        deterministic: bool = True,
        params: frozen_dict.FrozenDict,
        **kwargs,
    ) -> StepOutputs:
        """Forward pass the model and returns the output tree structure.

        Args:
            deterministic (bool): Whether to run the model in deterministic
                mode (e.g., disable dropout). Default is `True`.
            params (FrozenDict): The model parameters.
            **kwargs: Keyword arguments consumed by the model.

        Returns:
            The model outputs.
        """
        raise NotImplementedError
