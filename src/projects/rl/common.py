import abc
import dataclasses
import typing

import chex
import fiddle as fdl
import jax

from src.core import config as _config
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

    @abc.abstractmethod
    def get_action(
        self,
        *,
        obs: jax.Array,
        params: typing.Any,
        rngs: typing.Any,
        deterministic: bool = False,
        **kwargs,
    ) -> typing.Union[jax.Array, typing.Tuple[typing.Any, ...]]:
        r"""Execute the policy and returns the action.

        Args:
            obs (jax.Array): State observation array.
            params (typing.Any): Parameters for the policy.
            rngs (typing.Any): Random generator for reproducibility.
            deterministic (bool): Whether to run policy in deterministic mode.

        Returns:
            A single action array or a tuple of action and related outputs.
        """
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


@dataclasses.dataclass(frozen=False, kw_only=True)
class RLExperimentConfig:
    r"""Experiment configurations for running reinforcement learning algorithms.

    Attributes:
        project_name (str): Name of the project.
        exp_name (str): Name of the experiment.
        agent (fiddle.Partial): Partial function to instantiate the RL agent.
        environment (fiddle.Config): Build target for RL environment.
        trainer (TrainerConfig): Configurations for training.
        optimizer (OptimizerConfig): Configurations for the optimizers.
        replay_buffer_capacity (int): Number of samples in the replay buffer.
        warmup_steps (int): Random actions before training starts.
        n_envs (int): Number of parallel environments to run for on-policy RL.
        n_rollout_steps (int): Number of rollout steps per environment.
        gae_lambda (float): Discount factor in generalized advantage estimates.
        dtype (Any): The data type of the computation.
        param_dtype (Any): The data type of the parameters.
        precision (Any): Numerical precision of the computation.
        seed (int): Seed of random generator for reproducibility.
    """

    project_name: str
    exp_name: str

    # composed configuration objects
    agent: fdl.Partial[BaseAgent]
    environment: fdl.Config[BaseEnvironment]
    trainer: _config.TrainerConfig
    optimizer: _config.OptimizerConfig

    # replay buffer configurations
    replay_buffer_capacity: int = 1_000_000
    warmup_steps: int = 1_000

    # rollout configurations for on-policy RL
    n_envs: int = 1
    n_rollout_steps: int = 2_048
    gae_lambda: float = 0.95

    # global settings
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None
    seed: int = 42


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


__all__ = ["BaseAgent", "BaseEnvironment", "RLExperimentConfig", "StepTuple"]
