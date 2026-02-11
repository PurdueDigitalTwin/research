import typing

from jax import numpy as jnp
from jax import random as jrnd
import numpy as np

from src.projects.rl import structure


# Create a replay buffer class to store experiences
class ReplayBuffer:
    r"""Naive experience replay buffer."""

    def __init__(
        self,
        capacity: int,
        state_size: typing.Tuple[int, ...],
        action_size: typing.Tuple[int, ...] = (),
    ) -> None:
        r"""Initializes the replay buffer.

        Args:
            capacity (int, optional): Maximum number of experiences to store.
                Default is :math:`10000`.
        """
        self._capacity = capacity
        self._size: int = 0
        self._ptr: int = 0
        self._states = np.zeros([capacity, *state_size], dtype=np.float32)
        self._actions = np.zeros([capacity, *action_size], dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros([capacity, *state_size], dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        s: typing.Any,
        a: typing.Any,
        r: typing.Any,
        s_next: typing.Any,
        d: typing.Any = None,
    ) -> None:
        r"""Adds a new experience to the replay buffer.

        .. note::

            The buffer stores tuples of ``(s, a, r, s_next, d)``.
            Each element of the queue is a tuple.

        Args:
            s (Any): An array of the current environment state.
            a (Any): An array of the action taken.
            r (Any): A scalar array the reward received by taking the action.
            s_next (Any): An array the next state resulted from the action.
            d (Any): A boolean array whether the trajectory terminates.
        """
        self._states[self._ptr] = np.asarray(s, dtype=np.float32)
        self._actions[self._ptr] = np.asarray(a, dtype=np.float32)
        self._rewards[self._ptr] = np.asarray(r, dtype=np.float32)
        self._next_states[self._ptr] = np.asarray(s_next, dtype=np.float32)
        if d is None:
            d = False
        self._dones[self._ptr] = np.asarray(d, dtype=np.bool_)
        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, key: typing.Any) -> structure.StepTuple:
        r"""Samples a batch of experiences from the replay buffer.

        Args:
            batch_size (int): Number of experiences to sample.
            key (Any): Random key for reproducible sampling.

        Returns:
            A tuple of (states, actions, rewards, next_states, dones).
        """
        indices = jrnd.choice(key, self._size, (batch_size,), replace=False)
        # Convert JAX indices to a Python list of ints in one shot
        indices = indices.tolist()

        return structure.StepTuple(
            state=jnp.asarray(self._states[indices]),
            action=jnp.asarray(self._actions[indices]),
            reward=jnp.asarray(self._rewards[indices]),
            next_state=jnp.asarray(self._next_states[indices]),
            done=jnp.asarray(self._dones[indices]),
        )
