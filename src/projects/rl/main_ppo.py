# file created on Feb. 10, 2026
# The purpose of this file is for rl learning (PPO)
################################################
# framework: jax + flax linen
# environment: gym CartPole-v1
# reference: https://arxiv.org/pdf/1707.06347
# https://en.wikipedia.org/wiki/Policy_gradient_method#cite_note-3
# useful code repo: 
# https://github.com/ericyangyu/PPO-for-Beginners?tab=readme-ov-file
################################################
# Note: value-based methods struggle when the action space is large (continous 
# action space). Usually they are less efficient in terms of trainning time.
# Note: policy-based methods can learn stochastic policies where there isn't a 
# single best action, and they can also learn deterministic policies.
# Note: policy-based methods encourages exploration by nature.


import copy
import functools
import os
import typing

from absl import app
from absl import flags
from flax import linen as nn
from flax import serialization
import gymnasium as gym
import jax
from jax import lax
from jax import numpy as jnp
import jaxtyping
import matplotlib.pyplot as plt
import optax
import typing_extensions

from src.core import model as _model
from src.core import train_state as _train_state
from src.utilities import logging
from src.projects.rl import ppo
from src.projects.rl import structure as _structure
from src.projects.rl import policy
from src.projects.rl import replay_buffer as _replay_buffer


def train_step(
    model: _model.Model,
    train_state: _train_state.TrainState,
    batch: _structure.StepTuple,
) -> typing.Tuple[_train_state.TrainState, jax.Array]:
    r"""Performs a single training step for the PPO model.

    Args:
        model (Model): The PPO model to be trained.
        train_state (TrainState): The current training state containing model
            parameters and optimizer state.
        batch (StepTuple): A batch of experience tuples containing states, actions,
            rewards, next states, and done flags.
    Returns:
        Tuple[TrainState, jax.Array]: Updated training state and computed loss for
            the training step.
    """

    pass


def main(argv: typing.List[str]) -> None:
    r"""Main function to set up the environment, model, and training loop for PPO.

    Args:
        argv (List[str]): Command-line arguments (not used in this implementation).
    Returns:
        None.
    """
    del argv  # Unused.

    # Create the CartPole environment
    env = gym.make("CartPole-v1")

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
