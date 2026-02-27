import typing

from absl import app
from absl import flags
from flax import linen as nn
from flax import serialization
import gymnasium as gym
import jax

from src.projects.rl import replay_buffer
from src.utilities import logging

# Flags
flags.DEFINE_integer(
    name="buffer_capacity",
    default=30_000,
    help="Maximum number of transition tuples in a replay buffer.",
)
flags.DEFINE_integer(
    name="seed",
    default=42,
    help="Random generator seed.",
)


################################################################################
# Actor-critic Model
class ActorCriticNetwork(nn.Module):
    features: int
    out_features: int
    num_layers: int
    activation_fn: typing.Callable
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
    ) -> typing.Tuple[jax.Array, jax.Array]:
        raise NotImplementedError


class ActorCriticModel:
    def __init__(self) -> None:
        pass

    @property
    def network(self) -> ActorCriticNetwork:
        r"""ActorCriticNetwork: Backbone policy and critic networks."""
        return self._network


################################################################################
# Main entry point
def main(argv: typing.List[str]) -> None:
    del argv  # unused

    rngs = jax.random.PRNGKey(flags.FLAGS.seed)
    logging.rank_zero_info("Running with global seed %r", rngs)

    logging.rank_zero_info("Building the environment...")
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape or (1,)
    action_size = env.action_space.shape or (1,)
    logging.rank_zero_info(
        "Initialized environment %s with state size %r and action size %r.",
        env.__class__.__name__,
        state_size,
        action_size,
    )

    logging.rank_zero_info("Building replay buffer...")
    buffer = replay_buffer.ReplayBuffer(
        capacity=flags.FLAGS.buffer_capacity,
        state_size=env.observation_space.shape or (1,),
        action_size=env.action_space.shape or (1,),
    )
    logging.rank_zero_info("Successfully built %s.", buffer.__class__.__name__)

    env.close()


if __name__ == "__main__":
    app.run(main=main)
