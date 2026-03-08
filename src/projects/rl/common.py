import abc
import typing

import chex
import jax

from src.core import model as _model


class BaseAgent(_model.Model, abc.ABC):
    r"""Base class for reinforcement learning agents."""

    @abc.abstractmethod
    def configure_train_state(self, *args, **kwargs) -> typing.Any:
        r"""Configure and returns a training state container."""
        ...

    @property
    @abc.abstractmethod
    def is_on_policy(self) -> bool:
        r"""bool: Whether the agent runs an on-policy RL algorithm."""
        ...

    def on_train_batch_end(self, *, state: typing.Any, **kwargs) -> typing.Any:
        r"""Called at the end of a single training step."""
        return state  # NOTE: no-op by default.


class BaseEnvironment(abc.ABC):
    r"""Base class for reinforcement learning environment."""

    @abc.abstractmethod
    def step(self, action: typing.Any) -> typing.Any:
        r"""Take action and run one time step in the environment."""
        ...

    @abc.abstractmethod
    def reset(
        self,
        *,
        seed: typing.Optional[int] = None,
        **kwargs,
    ) -> typing.Any:
        r"""Resets the environment and returns an initial state."""
        ...

    def close(self) -> None:
        r"""Destructor for the environment."""
        pass  # NOTE: no-op by default


@chex.dataclass
class StepTuple:
    r"""Samples of a step in the environment ``(s,a,r,s')``.

    Attributes:
        state (Optional[jax.Array], optional): The current state array.
            Default is ``None``.
        action (Optional[jax.Array], optional): The action taken.
            Default is ``None``.
        reward (Optional[jax.Array], optional): Reward from taking the action.
            Default is ``None``.
        next_state (Optional[jax.Array], optional): Next state resulted from
            taking the action. Default is ``None``.
        done (Optional[jax.Array], optional): Whether the next state is a
            terminal state. Default is ``None``.
    """

    state: typing.Optional[jax.Array] = None
    action: typing.Optional[jax.Array] = None
    reward: typing.Optional[jax.Array] = None
    next_state: typing.Optional[jax.Array] = None
    done: typing.Optional[jax.Array] = None


__all__ = ["BaseAgent", "StepTuple"]
