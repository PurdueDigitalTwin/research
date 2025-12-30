import sys
import typing

import jax
from jax import numpy as jnp
import pytest

from src.projects.generative.model import unet


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_conv2d(dtype: typing.Any) -> None:
    r"""Tests the ported `Conv2D` layer."""
    rng = jax.random.PRNGKey(42)

    test_input = jax.random.normal(rng, (2, 2, 32, 32, 3), dtype=dtype)
    test_filter = [1, 3, 3, 1]

    # test Conv2D with downsampling only
    layer = unet.Conv2D(
        features=16,
        resample_filter=test_filter,
        downsampling=True,
        upsampling=False,
        dtype=dtype,
        param_dtype=dtype,
    )
    variables = layer.init(rngs={"params": rng}, inputs=test_input)
    outputs = layer.apply(variables=variables, inputs=test_input)
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (2, 2, 16, 16, 3)
    assert outputs.dtype == dtype

    # test Conv2D with upsampling only
    layer = unet.Conv2D(
        features=16,
        resample_filter=test_filter,
        downsampling=False,
        upsampling=True,
        dtype=dtype,
        param_dtype=dtype,
    )
    variables = layer.init(rngs={"params": rng}, inputs=test_input)
    outputs = layer.apply(variables=variables, inputs=test_input)
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (2, 2, 64, 64, 3)
    assert outputs.dtype == dtype

    # test Conv2D with convolution only
    layer = unet.Conv2D(
        features=16,
        kernel_size=1,
        use_bias=True,
        resample_filter=test_filter,
        downsampling=False,
        upsampling=False,
        dtype=dtype,
        param_dtype=dtype,
    )
    variables = layer.init(rngs={"params": rng}, inputs=test_input)
    assert "kernel" in variables["params"]["conv_out"]
    kernel = variables["params"]["conv_out"]["kernel"]
    assert isinstance(kernel, jax.Array)
    assert kernel.shape == (1, 1, 3, 16)
    bias = variables["params"]["conv_out"]["bias"]
    assert isinstance(bias, jax.Array)
    assert bias.shape == (16,)
    outputs = layer.apply(variables=variables, inputs=test_input)
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (2, 2, 32, 32, 16)
    assert outputs.dtype == dtype


@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_attn_block(num_heads: int, dtype: typing.Any) -> None:
    r"""Tests the attention block in U-Net models."""
    rng = jax.random.PRNGKey(42)

    block = unet.AttnBlock(
        num_heads=num_heads,
        num_groups=1,
        dtype=dtype,
        param_dtype=dtype,
    )
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
def test_edm_unet_block(dtype: typing.Any) -> None:
    r"""Tests the U-Net block for score-based generative modeling."""
    rng = jax.random.PRNGKey(42)

    # test downsampling block with attention
    block = unet.EDMUNetBlock(
        features=64,
        downsampling=True,
        dropout_rate=0.2,
        num_heads=1,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 2, 32, 32, 32), dtype=dtype)
    test_cond = jnp.ones((2, 2, 16), dtype=dtype)
    variables = block.init(
        rngs={"params": rng},
        inputs=test_input,
        cond=test_cond,
        deterministic=True,
    )
    outputs = block.apply(
        variables=variables,
        inputs=test_input,
        cond=test_cond,
        deterministic=True,
    )
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (2, 2, 16, 16, 64)
    assert outputs.dtype == dtype

    # test upsampling block without attention
    block = unet.EDMUNetBlock(
        features=64,
        upsampling=True,
        dropout_rate=0.2,
        num_heads=None,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 2, 16, 16, 64), dtype=dtype)
    test_cond = jnp.ones((2, 2, 16), dtype=dtype)
    variables = block.init(
        rngs={"params": rng},
        inputs=test_input,
        cond=test_cond,
        deterministic=True,
    )
    outputs = block.apply(
        variables=variables,
        inputs=test_input,
        cond=test_cond,
        deterministic=True,
    )
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (2, 2, 32, 32, 64)
    assert outputs.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_song_network(dtype: typing.Any) -> None:
    r"""Tests the full U-Net model for score-based generative modeling."""
    rng = jax.random.PRNGKey(42)

    model = unet.SongNetwork(
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
