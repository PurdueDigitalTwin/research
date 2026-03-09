import functools
import sys
import typing

from flax import jax_utils
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
import pytest

from src.projects.rl import common
from src.projects.rl.agents import dqn as _dqn

# ---------------------------------------------------------------------------
# Shared constants (CartPole-like environment)
# ---------------------------------------------------------------------------

rng = jax.random.PRNGKey(42)
OBS_DIM: int = 4
ACTION_DIM: int = 2
BATCH: int = 8
TARGET_UPDATE_FREQ: int = 10


# ---------------------------------------------------------------------------
# Helper: minimal Q-network (2-layer MLP)
# ---------------------------------------------------------------------------


class _QNetwork(nn.Module):
    r"""Minimal Q-network used in tests."""

    action_dim: int
    dtype: typing.Any
    param_dtype: typing.Any
    precision: typing.Any

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        del deterministic

        x = nn.Dense(
            64,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.action_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )(x)

        return x


def _make_agent(use_double: bool = False) -> _dqn.DQNModel:
    return _dqn.DQNModel(
        action_space_dim=ACTION_DIM,
        network=functools.partial(_QNetwork, action_dim=ACTION_DIM),
        gamma=0.99,
        q_target_update_freq=TARGET_UPDATE_FREQ,
        use_double=use_double,
    )


def _make_dummy_batch(
    leading: typing.Optional[int] = None,
) -> common.StepTuple:
    r"""Returns a StepTuple with ones; prepend a leading axis when given."""

    def _t(*shape: int, dtype: typing.Any = jnp.float32) -> jax.Array:
        return jnp.ones(shape, dtype=dtype)

    if leading is None:
        return common.StepTuple(
            state=_t(BATCH, OBS_DIM),
            action=jnp.zeros((BATCH, 1), dtype=jnp.int32),
            reward=_t(BATCH, 1),
            next_state=_t(BATCH, OBS_DIM),
            done=jnp.zeros((BATCH, 1)),
        )
    return common.StepTuple(
        state=_t(leading, BATCH, OBS_DIM),
        action=jnp.zeros((leading, BATCH, 1), dtype=jnp.int32),
        reward=_t(leading, BATCH),
        next_state=_t(leading, BATCH, OBS_DIM),
        done=jnp.zeros((leading, BATCH)),
    )


# ---------------------------------------------------------------------------
# Tests: properties
# ---------------------------------------------------------------------------


def test_dqn_is_not_on_policy() -> None:
    r"""DQN is an off-policy algorithm."""
    agent = _make_agent()
    assert agent.is_on_policy is False


# ---------------------------------------------------------------------------
# Tests: init
# ---------------------------------------------------------------------------


def test_dqn_init() -> None:
    r"""Init() returns a non-empty parameter pytree."""
    agent = _make_agent()
    batch = _make_dummy_batch()
    params, _ = agent.init(batch=batch, rngs=rng)

    leaves = jax.tree_util.tree_leaves(params)
    assert len(leaves) > 0, "params pytree should be non-empty"
    for leaf in leaves:
        assert isinstance(leaf, jax.Array)


# ---------------------------------------------------------------------------
# Tests: forward
# ---------------------------------------------------------------------------


def test_dqn_forward() -> None:
    r"""Forward() returns Q-values of shape (BATCH, ACTION_DIM)."""
    agent = _make_agent()
    batch = _make_dummy_batch()
    params, _ = agent.init(batch=batch, rngs=rng)

    outputs = agent.forward(batch=batch, params=params)

    assert outputs.output is not None
    assert isinstance(outputs.output, jax.Array)
    assert outputs.output.shape == (BATCH, ACTION_DIM)


# ---------------------------------------------------------------------------
# Tests: configure_train_state
# ---------------------------------------------------------------------------


def test_dqn_configure_train_state() -> None:
    r"""configure_train_state() creates a ``TrainState`` with target_params."""
    agent = _make_agent()
    batch = _make_dummy_batch()
    params, _ = agent.init(batch=batch, rngs=rng)
    tx = optax.adam(1e-4)
    state = agent.configure_train_state(params=params, tx=tx)

    assert state.step == 0
    assert state.mutables is not None
    assert "target_params" in state.mutables

    # target_params and params must share the same pytree structure
    params_leaves = jax.tree_util.tree_leaves(state.params)
    target_leaves = jax.tree_util.tree_leaves(state.mutables["target_params"])
    assert len(params_leaves) == len(target_leaves)
    for p, t in zip(params_leaves, target_leaves):
        assert p.shape == t.shape
        assert p.dtype == t.dtype


# ---------------------------------------------------------------------------
# Tests: training_step (requires jax.pmap)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_double", [False, True])
def test_dqn_training_step(use_double: bool) -> None:
    r"""training_step() decrements loss and increments step counter."""
    agent = _make_agent(use_double=use_double)
    batch = _make_dummy_batch()
    params, _ = agent.init(batch=batch, rngs=rng)
    tx = optax.adam(1e-4)
    state = agent.configure_train_state(params=params, tx=tx)

    n = jax.local_device_count()
    sharded_batch = _make_dummy_batch(leading=n)
    replicated_state = jax_utils.replicate(state)

    p_step = jax.pmap(
        functools.partial(agent.training_step, rngs=rng),
        axis_name="batch",
    )
    new_state, outputs = p_step(state=replicated_state, batch=sharded_batch)

    # step counter incremented on every device
    assert int(new_state.step[0]) == 1

    # loss is reported as a scalar per device
    assert outputs.scalars is not None
    loss = outputs.scalars["loss"]
    assert isinstance(loss, jax.Array)
    assert loss.shape == (n,)
    assert jnp.all(jnp.isfinite(loss))


# ---------------------------------------------------------------------------
# Tests: on_train_batch_end
# ---------------------------------------------------------------------------


def test_dqn_on_train_batch_end_no_sync() -> None:
    r"""on_train_batch_end NOT sync target params at non-divisible steps."""
    agent = _make_agent()
    batch = _make_dummy_batch()
    params, _ = agent.init(batch=batch, rngs=rng)
    tx = optax.adam(1e-4)
    state = agent.configure_train_state(params=params, tx=tx)

    # Perturb online params so they differ from target
    modified_params = jax.tree_util.tree_map(lambda x: x + 1.0, state.params)
    state = state.replace(params=modified_params)

    # Step not divisible by TARGET_UPDATE_FREQ → no sync expected
    step = TARGET_UPDATE_FREQ + 1
    new_state = agent.on_train_batch_end(state=state, step=step)

    target_leaves = jax.tree_util.tree_leaves(
        new_state.mutables["target_params"]
    )
    online_leaves = jax.tree_util.tree_leaves(new_state.params)
    for t, o in zip(target_leaves, online_leaves):
        # target should still differ from (modified) online params
        assert not jnp.allclose(
            t, o
        ), "Target params should NOT have been synced at a non-update step."


def test_dqn_on_train_batch_end_syncs_target() -> None:
    r"""on_train_batch_end syncs target params at divisible steps."""
    agent = _make_agent()
    batch = _make_dummy_batch()
    params, _ = agent.init(batch=batch, rngs=rng)
    tx = optax.adam(1e-4)
    state = agent.configure_train_state(params=params, tx=tx)

    # Perturb online params so they differ from initial target
    modified_params = jax.tree_util.tree_map(lambda x: x + 1.0, state.params)
    state = state.replace(params=modified_params)

    # Step divisible by TARGET_UPDATE_FREQ → sync expected
    new_state = agent.on_train_batch_end(state=state, step=TARGET_UPDATE_FREQ)

    target_leaves = jax.tree_util.tree_leaves(
        new_state.mutables["target_params"]
    )
    online_leaves = jax.tree_util.tree_leaves(new_state.params)
    for t, o in zip(target_leaves, online_leaves):
        assert jnp.allclose(
            t, o
        ), "Target params should equal online params after a sync step."


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
