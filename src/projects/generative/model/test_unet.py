import sys
import typing

import jax
from jax import numpy as jnp
import pytest

from src.projects.generative.model import unet


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_resnet_block(dtype: typing.Any) -> None:
    r"""Tests the residual downsampling block in U-Net models."""
    rng = jax.random.PRNGKey(42)

    block = unet.ResNetBlock(features=64, dtype=dtype, param_dtype=dtype)
    test_input = jnp.ones((2, 32, 32, 32), dtype=dtype)
    test_cond = jnp.ones((2, 16), dtype=dtype)
    params_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = block.init(
        rngs={"params": params_rng},
        inputs=test_input,
        cond=test_cond,
        deterministic=False,
    )
    outputs = block.apply(
        variables=variables,
        inputs=test_input,
        cond=test_cond,
        deterministic=False,
        rngs={"dropout": dropout_rng},
    )
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (2, 32, 32, 64)
    assert outputs.dtype == dtype


@pytest.mark.parametrize("with_conv", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_downsample_block(with_conv: bool, dtype: typing.Any) -> None:
    r"""Tests the downsampling block in U-Net models."""
    rng = jax.random.PRNGKey(42)

    block = unet.DownsampleBlock(
        with_conv=with_conv,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 32, 32, 32), dtype=dtype)
    variables = block.init(
        rngs={"params": rng},
        inputs=test_input,
    )
    if with_conv:
        assert "conv0" in variables["params"]
        kernel = variables["params"]["conv0"]["kernel"]
        assert isinstance(kernel, jax.Array)
        assert kernel.shape == (3, 3, 32, 32)
        bias = variables["params"]["conv0"]["bias"]
        assert isinstance(bias, jax.Array)
        assert bias.shape == (32,)

    outputs = block.apply(variables=variables, inputs=test_input)
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (2, 16, 16, 32)
    assert outputs.dtype == dtype


@pytest.mark.parametrize("with_conv", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_upsample_block(with_conv: bool, dtype: typing.Any) -> None:
    r"""Tests the upsampling block in U-Net models."""
    rng = jax.random.PRNGKey(42)

    block = unet.UpsampleBlock(
        with_conv=with_conv,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 16, 16, 32), dtype=dtype)
    variables = block.init(
        rngs={"params": rng},
        inputs=test_input,
    )
    if with_conv:
        assert "conv0" in variables["params"]
        kernel = variables["params"]["conv0"]["kernel"]
        assert isinstance(kernel, jax.Array)
        assert kernel.shape == (3, 3, 32, 32)
        bias = variables["params"]["conv0"]["bias"]
        assert isinstance(bias, jax.Array)
        assert bias.shape == (32,)

    outputs = block.apply(variables=variables, inputs=test_input)
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (2, 32, 32, 32)
    assert outputs.dtype == dtype


@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_attn_block(num_heads: int, dtype: typing.Any) -> None:
    r"""Tests the attention block in U-Net models."""
    rng = jax.random.PRNGKey(42)

    block = unet.AttnBlock(num_heads=num_heads, dtype=dtype, param_dtype=dtype)
    test_input = jnp.ones((2, 16, 16, 32), dtype=dtype)
    variables = block.init(
        rngs={"params": rng},
        inputs=test_input,
    )

    outputs = block.apply(variables=variables, inputs=test_input)
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (2, 16, 16, 32)
    assert outputs.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_score_net(dtype: typing.Any) -> None:
    r"""Tests the full U-Net model for score-based generative modeling."""
    rng = jax.random.PRNGKey(42)

    model = unet.ScoreNet(
        features=128,
        dropout_rate=0.2,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 32, 32, 3), dtype=dtype)
    test_cond = jnp.ones((2, 16), dtype=dtype)
    params_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = model.init(
        rngs={"params": params_rng},
        inputs=test_input,
        cond=test_cond,
        deterministic=True,
    )
    outputs = model.apply(
        variables=variables,
        inputs=test_input,
        cond=test_cond,
        deterministic=True,
        rngs={"dropout": dropout_rng},
    )
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (2, 32, 32, 3)
    assert outputs.dtype == dtype


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
