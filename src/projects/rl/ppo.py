# file created on Feb. 10, 2026
# The purpose of this file is for rl learning (PPO)
################################################
# framework: jax + flax linen
# environment: gym CartPole-v1
# reference: https://arxiv.org/pdf/1707.06347
# useful code repo: 
# https://github.com/ericyangyu/PPO-for-Beginners?tab=readme-ov-file
################################################
# Note: value-based methods struggle when the action space is large (continous 
# action space). Usually they are less efficient in terms of trainning time.
# Note: policy-based methods can learn stochastic policies where there isn't a 
# single best action, and they can also learn deterministic policies.
# Note: policy-based methods encourages exploration by nature.
# NOTE: PPO leans a stochastic policy, but we can't output multiple actions at
# the same time?


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
from src.projects.rl import structure as _struct
from src.projects.rl import dqn as _dqn
from src.projects.rl import replay_buffer as _buffer



