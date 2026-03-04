import abc
import typing

import chex
from flax.core import frozen_dict
import jax
import jaxtyping


@chex.dataclass
class StepOutputs:
    r"""A base container for outputs from a single step.

    Attributes:
        output (Optional[jax.Array]): The main output of the model.
        scalars (Optional[Dict[str, Any]]): A dictionary of scalar metrics.
        images (Optional[Dict[str, Any]]): A dictionary of image outputs.
        histograms (Optional[Dict[str, Array]]): A dictionary of array to
            plot as histograms.
    """

    output: typing.Optional[jax.Array] = None
    scalars: typing.Optional[typing.Dict[str, typing.Any]] = None
    images: typing.Optional[typing.Dict[str, typing.Any]] = None
    histograms: typing.Optional[typing.Dict[str, jax.Array]] = None


class Model(abc.ABC):
    r"""Interface for models."""

    @abc.abstractmethod
    def init(
        self,
        *,
        batch: typing.Any,
        rngs: typing.Union[typing.Any, typing.Dict[str, typing.Any]],
        **kwargs,
    ) -> typing.Tuple[jaxtyping.PyTree, jaxtyping.PyTree]:
        r"""Initializes the model parameters.

        Args:
            batch (Any): An example batch of data for initialization.
            rngs (Union[Any, Dict[str, Any]]): Random generators.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple of two pytree instances. The first contains initialized
                model parameters and the second are mutable variables such as
                the batch normalization parameters.
        """
        ...

    @abc.abstractmethod
    def forward(
        self,
        *,
        deterministic: bool = True,
        params: frozen_dict.FrozenDict,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Any:
        r"""Forward pass the model and returns the outputs.

        Args:
            deterministic (bool): Whether to run the model in deterministic
                mode (e.g., disable dropout). Default is `True`.
            params (FrozenDict): The model parameters.
            rngs (Any): Random key generator.

        Returns:
            The model outputs.
        """
        ...

    def training_step(
        self,
        *,
        batch: typing.Any,
        state: typing.Any,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Tuple[typing.Any, StepOutputs]:
        r"""Performs a single training step.

        Args:
            batch (Any): A batch of training data samples.
            state (Any): Data structure consisting of network parameters,
                mutable variables, and optimizer states.
            rngs (Any): Random key generator.

        Returns:
            A tuple of ``(new_state, step_outputs)``.
        """
        raise NotImplementedError("Train step function is not implemented.")

    def evaluation_step(
        self,
        *,
        batch: typing.Any,
        params: frozen_dict.FrozenDict,
        rngs: typing.Any,
        **kwargs,
    ) -> StepOutputs:
        r"""Performs a single evaluation step.

        Args:
            batch (Any): A batch of evaluation data samples.
            params (FrozenDict): The model parameters.
            rngs (Any): Random key generator.

        Returns:
            The evaluation step outputs.
        """
        raise NotImplementedError("Evaluate step function is not implemented.")


__all__ = ["StepOutputs", "Model"]
