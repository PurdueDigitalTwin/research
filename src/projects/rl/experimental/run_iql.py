# file created on Apr. 3rd, 2026 by Yaguang Li
# Implicit Q-Learning (IQL) implementation for offline RL.
################################################
# framework: jax + flax linen
# environment
# Reference: https://arxiv.org/abs/2110.06169
################################################


import copy
import functools
import os
import typing

from absl import app
from absl import flags
from flax import jax_utils
from flax import serialization
import gymnasium as gym
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import optax
import minari

from src.core import train_state as _train_state
from src.projects.rl import dqn as _dqn
from src.projects.rl import replay_buffer as _buffer
from src.projects.rl import structure as _struct
from src.utilities import logging
from src.utilities import training


def main(argv: typing.List[str]) -> None:
    del argv  # Unused.

    # NOTE: refer to minari documentation on the difference between simple,
    # medium, and expert datasets.
    dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0")

    


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
