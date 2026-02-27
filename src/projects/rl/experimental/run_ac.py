import collections
import functools
import typing

from absl import app
from absl import flags
from flax import jax_utils
from flax import linen as nn
from flax.training import common_utils
import gymnasium as gym
import jax
from jax import numpy as jnp
import jaxtyping
import optax

from src.core import model as _model
from src.core import train_state as _train_state
from src.projects.rl import replay_buffer as _replay_buffer
from src.projects.rl import structure as _structure
from src.utilities import logging as _logging

# Flags
flags.DEFINE_integer(
    name="batch_size",
    default=1024,
    help="Number of transition tuples for single training step.",
)
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
    name="num_episodes",
    default=1_000,
    help="Number of episodes to train the Actor-Critic model.",
)
flags.DEFINE_integer(
    name="eval_every_n_episodes",
    default=50,
    required=False,
    help="Evaluation frequency (in episodes) during training.",
)
flags.DEFINE_integer(
    name="seed",
    default=42,
    help="Random generator seed.",
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    required=True,
    help="Working directory",
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

        variables = self.network.init(rngs, state)
        if jax.process_index() == 0:
            _tabulate_fn = nn.summary.tabulate(
                module=self.network,
                rngs=rngs,
                depth=2,
                console_kwargs=dict(width=120, force_jupyter=False),
            )
            print(_tabulate_fn(state))

        return variables["params"]

    def training_step(
        self,
        *,
        batch: _structure.StepTuple,
        state: _train_state.TrainState,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Tuple[_train_state.TrainState, _model.StepOutputs]:
        del kwargs

        local_rng = jax.random.fold_in(rngs, jax.lax.axis_index("batch"))
        local_rng = jax.random.fold_in(local_rng, state.step)

        # step 1: compute TD-error and update the Q-function
        def critic_loss_fn(params: jaxtyping.PyTree) -> jax.Array:
            if batch.action is None:
                raise ValueError("Action is required for Q-learning.")
            outputs = self.network.apply(
                variables={"params": params},
                state=batch.state,
                rngs=local_rng,
            )
            assert isinstance(outputs, typing.Dict)
            q_vals = outputs["critic"]["q_values"]
            one_hot_a = jax.nn.one_hot(
                batch.action[..., 0].astype(jnp.int32),
                num_classes=q_vals.shape[-1],
            )
            q_vals = jnp.sum(q_vals * jax.lax.stop_gradient(one_hot_a), -1)

            outputs = self.network.apply(
                variables={"params": params},
                state=batch.next_state,
                rngs=local_rng,
            )
            assert isinstance(outputs, typing.Dict)
            prob_actions = jax.nn.softmax(outputs["actor"]["logits"])
            q_next = jnp.sum(
                prob_actions * outputs["critic"]["q_values"],
                axis=-1,
            )

            if batch.reward is None:
                r = jnp.zeros_like(q_next)
            else:
                r = batch.reward.astype(q_next)

            if batch.done is None:
                d = jnp.zeros_like(q_next)
            else:
                d = batch.done.astype(q_next.dtype)

            q_tgt = jax.lax.stop_gradient(r + self._gamma * q_next * (1 - d))
            loss = optax.squared_error(q_vals, q_tgt).mean()

            return loss

        grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=False)
        critic_loss, critic_grads = grad_fn(state.params)
        critic_loss = jax.lax.pmean(critic_loss, axis_name="batch")
        critic_grads = jax.lax.pmean(critic_grads, axis_name="batch")
        new_state = state.apply_gradients(grads=critic_grads)
        new_state = new_state.replace(step=state.step)

        # step 2: compute the policy loss and update the policy function
        def actor_loss_fn(params: jaxtyping.PyTree) -> jax.Array:
            outputs = self.network.apply(
                variables={"params": params},
                state=batch.state,
                rngs=local_rng,
            )
            assert isinstance(outputs, typing.Dict)
            q_vals = jax.lax.stop_gradient(outputs["critic"]["q_values"])
            logits = outputs["actor"]["logits"]

            loss = jnp.sum(
                -jax.nn.softmax(logits, axis=-1) * q_vals,
                axis=-1,
            ).mean()

            return loss

        grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=False)
        actor_loss, actor_grads = grad_fn(new_state.params)
        actor_loss = jax.lax.pmean(actor_loss, axis_name="batch")
        actor_grads = jax.lax.pmean(actor_grads, axis_name="batch")
        new_state = new_state.apply_gradients(grads=actor_grads)

        outputs = _model.StepOutputs(
            scalars=dict(
                actor_loss=actor_loss.mean(),
                actor_grad_norm=optax.global_norm(actor_grads).mean(),
                critic_loss=critic_loss.mean(),
                critic_grad_norm=optax.global_norm(critic_grads).mean(),
            )
        )

        return new_state, outputs

    def evaluation_step(
        self,
        *,
        batch: jaxtyping.PyTree,
        params: jaxtyping.PyTree,
        rngs: typing.Any,
        **kwargs,
    ) -> _model.StepOutputs:
        raise NotImplementedError

    def predict_step(
        self,
        *,
        batch: jaxtyping.PyTree,
        params: jaxtyping.PyTree,
        rngs: typing.Any,
        **kwargs,
    ) -> jax.Array:
        del kwargs

        outputs = self.network.apply(
            variables={"params": params},
            state=batch,
            rngs=rngs,
        )
        assert isinstance(outputs, typing.Dict)
        logits = outputs["actor"]["logits"]

        return jnp.argmax(logits, axis=-1)


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

    rngs, train_key, eval_key, predict_key = jax.random.split(rngs, num=4)
    p_train_step = functools.partial(model.training_step, rngs=train_key)
    p_train_step = jax.pmap(p_train_step, axis_name="batch")
    p_eval_step = functools.partial(model.evaluation_step, rngs=eval_key)
    p_eval_step = jax.pmap(p_eval_step, axis_name="batch")
    p_predict_step = functools.partial(model.predict_step, rngs=predict_key)
    p_predict_step = jax.pmap(p_predict_step, axis_name="batch")

    train_step_cntr: int = int(train_state.step)
    train_state: _train_state.TrainState = jax_utils.replicate(train_state)
    try:
        _logging.rank_zero_info("Populating the replay buffer...")
        state, _ = env.reset()
        for _ in range(flags.FLAGS.buffer_capacity):
            batch = jax.tree_util.tree_map(common_utils.shard, state)
            action = p_predict_step(batch=batch, params=train_state.params)
            action = jax.device_get(action).reshape(-1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, done)
            state = next_state

            if done:
                state, _ = env.reset()
        _logging.rank_zero_info("Populating the replay buffer... DONE")

        # main training loop
        _logging.rank_zero_info("Training...")
        train_scalars = collections.defaultdict(list)
        eval_scalars = collections.defaultdict(list)
        rngs, sample_key = jax.random.split(rngs, num=2)
        for episode in range(1, flags.FLAGS.num_episodes + 1):
            if episode % flags.FLAGS.eval_every_n_episodes == 0:
                _logging.rank_zero_info(f"Evaluating at episode {episode:d}")
                eval_env = gym.make("CartPole-v1")
                eval_rewards = []
                for _ in range(5):
                    state, _ = eval_env.reset()
                    done = False
                    total_reward = 0

                    while not done:
                        batch = jax.tree_util.tree_map(
                            common_utils.shard, state
                        )
                        action = p_predict_step(
                            batch=batch,
                            params=train_state.params,
                        )
                        action = jax.device_get(action).reshape(-1).item()
                        (
                            state,
                            reward,
                            terminated,
                            truncated,
                            _,
                        ) = eval_env.step(action)
                        done = terminated or truncated
                        total_reward += float(reward)

                    eval_rewards.append(total_reward)
                eval_env.close()
                _logging.rank_zero_info(
                    f"Eval at episode {episode:d} | "
                    f"Min Reward = {min(eval_rewards)} | "
                    f"Max Reward = {max(eval_rewards)} | "
                    f"Average Reward = {sum(eval_rewards) / len(eval_rewards)}."
                )
                eval_scalars["eval_min_reward"].append(min(eval_rewards))
                eval_scalars["eval_max_reward"].append(max(eval_rewards))
                eval_scalars["eval_avg_reward"].append(
                    sum(eval_rewards) / len(eval_rewards)
                )
                _logging.rank_zero_info("Evaluation DONE!")

            state, _ = env.reset()
            done = False
            episode_scalars = collections.defaultdict(list)

            while not done:
                # sample a batch of transition tuples and train
                sample_key = jax.random.fold_in(sample_key, train_step_cntr)
                batch = buffer.sample(flags.FLAGS.batch_size, sample_key)
                assert isinstance(batch, _structure.StepTuple)
                leaves, tree_def = jax.tree_util.tree_flatten(batch)
                leaves = jax.tree_util.tree_map(common_utils.shard, leaves)
                batch = jax.tree_util.tree_unflatten(tree_def, leaves)
                train_state, outputs = p_train_step(
                    batch=batch,
                    state=train_state,
                )
                assert isinstance(outputs, _model.StepOutputs)
                if outputs.scalars is not None:
                    scalar_str = f"Step: {train_step_cntr:d} |"
                    for k, v in outputs.scalars.items():
                        v = jax.device_get(v).item()
                        train_scalars[f"train_{k}_step"].append(v)
                        episode_scalars[k].append(v)
                        scalar_str += f" {k.capitalize()}: {v:.4f} |"
                    if train_step_cntr % 50 == 0:
                        _logging.rank_zero_info(scalar_str)

                # execute the new policy and save transition
                batch = jax.tree_util.tree_map(common_utils.shard, state)
                action = p_predict_step(batch=batch, params=train_state.params)
                action = jax.device_get(action).reshape(-1).item()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.add(state, action, reward, next_state, done)
                state = next_state

                train_step_cntr += 1

            for k, v in episode_scalars.items():
                train_scalars[f"train_{k}_episode"].append(sum(v) / len(v))

    finally:
        train_state = jax_utils.unreplicate(train_state)
        env.close()

    env.close()


if __name__ == "__main__":
    app.run(main=main)
