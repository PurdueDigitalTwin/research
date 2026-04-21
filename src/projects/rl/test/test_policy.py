import sys
import typing

import flax.linen as nn
import jax
from jax import numpy as jnp
import pytest

from src.projects.rl import policy

rng = jax.random.PRNGKey(42)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_layers", [1, 4])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_mlp_policy_output_shape(
    batch_size: int,
    num_layers: int,
    dtype: typing.Any,
) -> None:
    r"""Tests the forward shape and dtype of ``MlpPolicy``."""
    model = policy.MlpPolicy(
        features=16,
        out_features=5,
        num_layers=num_layers,
        activation=nn.gelu,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((batch_size, 8), dtype=dtype)
    params = model.init(rng, test_input)
    outputs = model.apply(params, test_input)

    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (batch_size, 5)
    assert outputs.dtype == dtype


def test_mlp_policy_param_structure() -> None:
    r"""Tests that ``MlpPolicy`` creates ``fc_1``..``fc_N`` dense layers."""
    num_layers = 3
    model = policy.MlpPolicy(
        features=16,
        out_features=4,
        num_layers=num_layers,
        activation=nn.relu,
    )
    params = model.init(rng, jnp.ones((2, 8)))

    for i in range(1, num_layers + 1):
        assert f"fc_{i}" in params["params"]

    # The last layer should project to ``out_features``.
    last_kernel = params["params"][f"fc_{num_layers}"]["kernel"]
    assert isinstance(last_kernel, jax.Array)
    assert last_kernel.shape == (16, 4)


def test_mlp_policy_single_layer_is_linear() -> None:
    r"""A one-layer MLP should apply no activation function."""
    model = policy.MlpPolicy(
        features=16,
        out_features=3,
        num_layers=1,
        activation=nn.relu,  # should be a no-op for a single layer
    )
    test_input = jnp.array([[-1.0, -1.0, -1.0]])
    params = model.init(rng, test_input)

    # Force negative pre-activations by setting bias to -5.0.
    biased_params = jax.tree_util.tree_map(lambda x: x, params)
    biased_params["params"]["fc_1"]["bias"] = jnp.full((3,), -5.0)
    outputs = model.apply(biased_params, test_input)

    # NOTE: If relu were applied, outputs would be clamped to 0. Since no
    # activation is used on the final layer, outputs must be negative values.
    assert isinstance(outputs, jax.Array)
    assert jnp.any(outputs < 0.0)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_gaussian_policy_output_shape(
    batch_size: int,
    dtype: typing.Any,
) -> None:
    r"""``GaussianPolicy`` returns ``(mean, log_std)`` with matching shapes."""
    action_dim = 6
    model = policy.GaussianPolicy(
        features=16,
        out_features=action_dim,
        num_layers=3,
        activation=nn.gelu,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((batch_size, 10), dtype=dtype)
    params = model.init(rng, test_input)
    mean, log_std = model.apply(params, test_input)

    assert isinstance(mean, jax.Array)
    assert isinstance(log_std, jax.Array)
    assert mean.shape == (batch_size, action_dim)
    assert log_std.shape == (batch_size, action_dim)
    assert mean.dtype == dtype
    assert log_std.dtype == dtype


def test_gaussian_policy_state_independent_log_std() -> None:
    r"""When ``use_state_dependent_std=False``, ``log_std`` is a single param broadcast across the
    batch and is independent of the input."""
    model = policy.GaussianPolicy(
        features=8,
        out_features=3,
        num_layers=2,
        activation=nn.relu,
        use_state_dependent_std=False,
    )
    params = model.init(rng, jnp.ones((2, 4)))

    # A free ``log_std`` parameter should exist at the top level.
    assert "log_std" in params["params"]
    log_std = params["params"]["log_std"]
    assert isinstance(log_std, jax.Array)
    assert log_std.shape == (3,)

    # ``log_std`` values must match for any two different inputs.
    _, log_std_a = model.apply(params, jnp.ones((1, 4)))
    _, log_std_b = model.apply(params, jnp.full((1, 4), 10.0))
    assert isinstance(log_std_a, jax.Array)
    assert isinstance(log_std_b, jax.Array)
    assert jnp.allclose(log_std_a, log_std_b)


def test_gaussian_policy_state_dependent_log_std_is_clipped() -> None:
    r"""When ``use_state_dependent_std=True``, ``log_std`` should be clipped to ``[-5.0, 2.0]`` for
    numerical stability."""
    model = policy.GaussianPolicy(
        features=8,
        out_features=3,
        num_layers=2,
        activation=nn.relu,
        use_state_dependent_std=True,
    )
    # Wide range of inputs to provoke large pre-clip values.
    test_input = jnp.linspace(-1e3, 1e3, 24).reshape((6, 4))
    params = model.init(rng, test_input)
    _, log_std = model.apply(params, test_input)
    assert isinstance(log_std, jax.Array)
    assert jnp.all(log_std >= -5.0)
    assert jnp.all(log_std <= 2.0)


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
