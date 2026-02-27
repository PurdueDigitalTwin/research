import typing

from absl import app
from absl import flags
from flax import linen as nn
from flax import serialization
import gymnasium as gym
import jax
from jax import numpy as jnp
import jaxtyping
import optax

from src.core import train_state as _train_state
from src.projects.rl import replay_buffer as _replay_buffer
from src.utilities import logging as _logging

# Flags
flags.DEFINE_integer(
    name="buffer_capacity",
    default=30_000,
    help="Maximum number of transition tuples in a replay buffer.",
)
flags.DEFINE_float(
    name="gamma",
    default=0.99,
    help="Discount factor for future rewards in the reinforcement learning.",
)
flags.DEFINE_integer(
    name="seed",
    default=42,
    help="Random generator seed.",
)


################################################################################
# Actor-critic Model
class ActorCriticNetwork(nn.Module):
    r"""Backbone network for actor-critic model with shared backbone."""

    features: int
    out_features: int
    num_layers: int
    activation_fn: typing.Callable
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, state: jax.Array) -> typing.Dict[str, jaxtyping.PyTree]:
        r"""Forward pass the actor and critic networks.

        Args:
            state (jax.Array): Observed state of shape `(*, state_size)`.

        Returns:
            A dictionary of actor and critic outputs, where each value is a
                `jaxtyping.PyTree`. For example, the actor output can be a
                a single logit array for discrete action space, and the critic
                output will then be the Q-function values of each action.
        """
        out = state.astype(self.dtype)

        scale = self.features ** (-0.5)
        for i in range(self.num_layers - 1):
            out = nn.Dense(
                features=self.features,
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=scale,
                    mode="fan_avg",
                    distribution="normal",
                ),
                use_bias=True,
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"linear_{i+1:d}",
            )(out)
            out = self.activation_fn(out)

        logits = nn.Dense(
            features=self.out_features,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1e-10,
                mode="fan_avg",
                distribution="normal",
            ),
            use_bias=True,
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="actor_head",
        )(out)

        q_values = nn.Dense(
            features=self.out_features,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1e-10,
                mode="fan_avg",
                distribution="normal",
            ),
            use_bias=True,
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="critic_head",
        )(out)

        return {"actor": {"logits": logits}, "critic": {"q_values": q_values}}


class ActorCriticModel:
    r"""Actor-critic model with."""

    def __init__(
        self,
        action_space_dim: int,
        gamma: float,
    ) -> None:
        self._action_space_dim = action_space_dim
        self._gamma = gamma
        self._network = ActorCriticNetwork(
            features=64,
            out_features=action_space_dim,
            num_layers=2,
            activation_fn=jax.nn.tanh,
        )
        pass

    @property
    def network(self) -> ActorCriticNetwork:
        r"""ActorCriticNetwork: Backbone policy and critic networks."""
        return self._network

    def init(
        self,
        *,
        state: jax.Array,
        rngs: typing.Any,
        **kwargs,
    ) -> jaxtyping.PyTree:
        del kwargs

        params = self.network.init(rngs, state)
        if jax.process_index() == 0:
            _tabulate_fn = nn.summary.tabulate(
                module=self.network,
                rngs=rngs,
                depth=2,
                console_kwargs=dict(width=120, force_jupyter=False),
            )
            print(_tabulate_fn(state))

        return params


################################################################################
# Main entry point
def main(argv: typing.List[str]) -> None:
    del argv  # unused

    rngs = jax.random.PRNGKey(flags.FLAGS.seed)
    _logging.rank_zero_info("Running with global seed %r", rngs)

    _logging.rank_zero_info("Building the environment...")
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape or (1,)
    action_size = env.action_space.shape or (1,)
    _logging.rank_zero_info(
        "Initialized environment %s with state size %r and action size %r.",
        env.__class__.__name__,
        state_size,
        action_size,
    )

    _logging.rank_zero_info("Building replay buffer...")
    buffer = _replay_buffer.ReplayBuffer(
        capacity=flags.FLAGS.buffer_capacity,
        state_size=env.observation_space.shape or (1,),
        action_size=env.action_space.shape or (1,),
    )
    _logging.rank_zero_info(
        "Successfully built %s.",
        buffer.__class__.__name__,
    )

    _logging.rank_zero_info("Building an actor-critic model.")
    rngs, init_key = jax.random.split(rngs, num=2)
    model = ActorCriticModel(
        action_space_dim=env.action_space.n,  # type: ignore
        gamma=flags.FLAGS.gamma,
    )
    params = model.init(state=jnp.zeros((1, *state_size)), rngs=init_key)
    _logging.rank_zero_info("Successfully built %s", model.__class__.__name__)

    _logging.rank_zero_info("Building training state...")
    lr_scheduler = optax.warmup_constant_schedule(0.0, 1e-4, 2_000)
    train_state = _train_state.TrainState.create(
        params=params,
        tx=optax.adam(lr_scheduler),
        ema_rate=0.0,  # NOTE: do not apply exponential moving average
    )
    jax.block_until_ready(train_state)
    _logging.rank_zero_info(
        "Successfully built %s",
        train_state.__class__.__name__,
    )

    env.close()


if __name__ == "__main__":
    app.run(main=main)
