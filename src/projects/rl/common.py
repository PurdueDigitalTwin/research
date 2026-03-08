import typing

import chex
import jax


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
