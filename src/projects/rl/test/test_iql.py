import copy
import sys
import typing

import jax
from jax import numpy as jnp
import optax
import pytest

from src.core import model as _model
from src.core import train_state as _train_state
from src.projects.rl import iql
from src.projects.rl import structure as _struct

rng = jax.random.PRNGKey(42)

STATE_DIM = 5
ACTION_DIM = 3
BATCH_SIZE = 4


def _make_agent(
    tau: float = 0.7, gamma: float = 0.99, beta: float = 3.0
) -> iql.IQLModel:
    return iql.IQLModel(
        action_space_dim=ACTION_DIM,
        tau=tau,
        gamma=gamma,
        beta=beta,
    )


def _make_batch(
    batch_size: int = BATCH_SIZE,
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
) -> _struct.StepTuple:
    r"""Builds a dummy batch whose shapes satisfy all internal assertions."""
    key = jax.random.PRNGKey(0)
    k_s, k_a, k_r, k_sp = jax.random.split(key, 4)
    return _struct.StepTuple(
        state=jax.random.normal(k_s, (batch_size, state_dim)),
        action=jax.random.uniform(
            k_a, (batch_size, action_dim), minval=-1.0, maxval=1.0
        ),
        reward=jax.random.normal(k_r, (batch_size,)),
        next_state=jax.random.normal(k_sp, (batch_size, state_dim)),
        done=jnp.zeros((batch_size,), dtype=jnp.float32),
    )


def _make_train_states(
    agent: iql.IQLModel,
    batch: _struct.StepTuple,
    learning_rate: float = 1e-3,
) -> typing.Tuple[
    typing.Any,
    typing.Tuple[_train_state.TrainState, ...],
    typing.Any,
]:
    r"""Initializes IQL params and wraps them in per-network TrainStates."""
    v_params, q_params, p_params = agent.init(batch=batch, rngs=rng)
    tx = optax.adam(learning_rate=learning_rate)
    v_state = _train_state.TrainState.create(params=v_params, tx=tx)
    q_state = _train_state.TrainState.create(params=q_params, tx=tx)
    p_state = _train_state.TrainState.create(params=p_params, tx=tx)
    target_params = copy.deepcopy(q_params)
    return (
        (v_params, q_params, p_params),
        (v_state, q_state, p_state),
        target_params,
    )


def test_iql_init_returns_three_param_trees() -> None:
    r"""``init`` returns ``(value_params, q_params, policy_params)`` where ``q_params`` is a
    2-tuple for clipped double Q-learning."""
    agent = _make_agent()
    batch = _make_batch()
    v_params, q_params, p_params = agent.init(batch=batch, rngs=rng)

    assert hasattr(v_params, "keys")
    assert hasattr(p_params, "keys")
    assert isinstance(q_params, tuple) and len(q_params) == 2

    # The two Q-networks must have the same pytree structure but be initialized
    # from different rngs, so at least some leaves must differ.
    q1_leaves = jax.tree_util.tree_leaves(q_params[0])
    q2_leaves = jax.tree_util.tree_leaves(q_params[1])
    assert len(q1_leaves) == len(q2_leaves)
    assert any(not jnp.allclose(a, b) for a, b in zip(q1_leaves, q2_leaves))


def test_iql_init_raises_when_action_is_missing() -> None:
    r"""``init`` must raise when ``batch.action`` is missing."""
    agent = _make_agent()
    bad_batch = _struct.StepTuple(
        state=jnp.ones((1, STATE_DIM)),
        action=None,
    )
    with pytest.raises(ValueError):
        agent.init(batch=bad_batch, rngs=rng)


@pytest.mark.parametrize("tau", [0.5, 0.7, 0.9])
def test_expectile_loss_weighting(tau: float) -> None:
    r"""Checks the asymmetric weighting of the expectile loss.

    .. note::
        For ``diff = target - value > 0``, the weight is ``tau``; otherwise it
        is ``1 - tau``. At ``tau == 0.5`` the loss reduces to squared error.
    """
    agent = _make_agent(tau=tau)
    value = jnp.array([0.0, 0.0])
    target = jnp.array([1.0, -1.0])  # positive diff, negative diff

    loss = agent._expectile_loss(value=value, target=target)

    expected = jnp.array([tau * 1.0, (1.0 - tau) * 1.0])
    assert jnp.allclose(loss, expected, atol=1e-6)


def test_expectile_loss_symmetric_at_half() -> None:
    r"""At ``tau=0.5`` the expectile loss reduces to ``0.5 * (diff)^2``."""
    agent = _make_agent(tau=0.5)
    value = jnp.array([0.3, -0.2, 1.5])
    target = jnp.array([1.0, -1.5, 0.5])
    loss = agent._expectile_loss(value=value, target=target)
    assert jnp.allclose(loss, 0.5 * (target - value) ** 2, atol=1e-6)


def test_iql_forward_output_shapes() -> None:
    r"""``forward`` returns ``(value, q_min, policy_out)`` where ``q_min`` is the element-wise
    minimum of the two Q-networks and ``policy_out`` is the ``(mean, log_std)`` tuple produced by
    the Gaussian policy."""
    agent = _make_agent()
    batch = _make_batch()
    params, _, _ = _make_train_states(agent, batch)

    value, q_min, policy_out = agent.forward(params=params, batch=batch)

    assert isinstance(value, jax.Array)
    assert value.shape == (BATCH_SIZE, 1)
    assert isinstance(q_min, jax.Array)
    assert q_min.shape == (BATCH_SIZE, 1)

    mean, log_std = policy_out
    assert mean.shape == (BATCH_SIZE, ACTION_DIM)
    assert log_std.shape == (BATCH_SIZE, ACTION_DIM)

    # ``q_min`` must equal the element-wise min of the two Q-network outputs.
    _, q_params, _ = params
    q_input = jnp.concatenate([batch.state, batch.action], axis=-1)
    q1 = agent._q_network.apply(q_params[0], q_input)
    q2 = agent._q_network.apply(q_params[1], q_input)
    assert jnp.allclose(q_min, jnp.minimum(q1, q2))


def test_training_v_step_updates_state_and_returns_scalar_loss() -> None:
    r"""A single value-network update advances ``step`` and emits a scalar."""
    agent = _make_agent()
    batch = _make_batch()
    params, (v_state, _, _), target_params = _make_train_states(agent, batch)

    new_v_state, outputs = agent.training_v_step(
        params=params,
        batch=batch,
        train_state=v_state,
        target_params=target_params,
        rngs=rng,
    )

    assert isinstance(outputs, _model.StepOutputs)
    assert outputs.scalars is not None
    assert "value_loss" in outputs.scalars
    assert jnp.isfinite(outputs.scalars["value_loss"])
    assert new_v_state.step == v_state.step + 1

    # Parameters must actually have been updated.
    leaves_before = jax.tree_util.tree_leaves(v_state.params)
    leaves_after = jax.tree_util.tree_leaves(new_v_state.params)
    assert any(
        not jnp.allclose(a, b) for a, b in zip(leaves_before, leaves_after)
    )


def test_training_q_step_updates_state_and_returns_scalar_loss() -> None:
    r"""A single Q-network update advances ``step`` and emits a scalar."""
    agent = _make_agent()
    batch = _make_batch()
    params, (_, q_state, _), target_params = _make_train_states(agent, batch)

    new_q_state, outputs = agent.training_q_step(
        params=params,
        batch=batch,
        train_state=q_state,
        target_params=target_params,
        rngs=rng,
    )

    assert isinstance(outputs, _model.StepOutputs)
    assert outputs.scalars is not None
    assert "q_loss" in outputs.scalars
    assert jnp.isfinite(outputs.scalars["q_loss"])
    assert new_q_state.step == q_state.step + 1


def test_training_p_step_updates_state_and_returns_scalar_loss() -> None:
    r"""A single policy update advances ``step`` and emits a scalar."""
    agent = _make_agent()
    batch = _make_batch()
    params, (_, _, p_state), target_params = _make_train_states(agent, batch)

    new_p_state, outputs = agent.training_p_step(
        params=params,
        batch=batch,
        train_state=p_state,
        target_params=target_params,
        rngs=rng,
    )

    assert isinstance(outputs, _model.StepOutputs)
    assert outputs.scalars is not None
    assert "policy_loss" in outputs.scalars
    assert jnp.isfinite(outputs.scalars["policy_loss"])
    assert new_p_state.step == p_state.step + 1


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
