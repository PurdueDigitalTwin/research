import collections
import random

import jax.numpy as jnp

from src.projects.rl import structure


# Create a replay buffer class to store experiences
class ReplayBuffer:
    def __init__(self, capacity=10000) -> None:
        r"""Initializes the replay buffer.

        Args:
            capacity (int): Maximum number of experiences to store.

        Returns:
            None.
        """
        self.buffer = collections.deque(
            maxlen=capacity
        )  # deque is a double-ended queue.

    def add(self, s, a, r, s_next, d) -> None:
        r"""Adds a new experience to the replay buffer. The buffer stores tuples of (s, a, r,
        s_next, d). Each element of the queue is a tuple.

        Args:
            s: state
            a: action
            r: reward
            s_next: next state
            d: done flag

        Returns:
            None.
        """
        if d is None:
            d = False
        self.buffer.append((s, a, r, s_next, d))

    def sample(self, batch_size) -> structure.StepTuple:
        r"""Samples a batch of experiences from the replay buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            A tuple of (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(
            self.buffer, batch_size
        )  # randomly sample batch_size number of experiences from the buffer
        s, a, r, s_next, d = zip(*batch)  # unzip the batch into separate lists

        return structure.StepTuple(
            state=jnp.array(s),
            action=jnp.array(a),
            reward=jnp.array(r),
            next_state=jnp.array(s_next),
            done=jnp.array(d),
        )
