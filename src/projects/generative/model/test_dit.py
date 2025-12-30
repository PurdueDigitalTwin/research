import sys
import typing

import jax
from jax import numpy as jnp
import pytest

from src.projects.generative.model import dit


@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_attention(num_heads: int, dtype: typing.Any) -> None:
    r"""Test the multi-head attention module for Diffusion Transformer."""
    layer = dit.Attention(
        features=16,
        num_heads=num_heads,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 4, 16), dtype=dtype)
    test_cond = jnp.ones((2, 1, 16), dtype=dtype)

    # test self-attention
    variables = layer.init(
        jax.random.PRNGKey(0),
        test_input,
        deterministic=True,
    )
    test_output = layer.apply(
        variables,
        test_input,
        deterministic=False,
        rngs={"dropout": jax.random.PRNGKey(1)},
    )
    assert isinstance(test_output, jax.Array)
    assert test_output.shape == (2, 4, 16)
    assert test_output.dtype == dtype

    # test cross-attention
    variables = layer.init(
        jax.random.PRNGKey(0),
        test_input,
        key_value=test_cond,
        deterministic=True,
    )
    test_output = layer.apply(
        variables,
        test_input,
        key_value=test_cond,
        deterministic=False,
        rngs={"dropout": jax.random.PRNGKey(1)},
    )
    assert isinstance(test_output, jax.Array)
    assert test_output.shape == (2, 4, 16)
    assert test_output.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_in_context_dit_block(dtype: typing.Any) -> None:
    r"""Test the DiT block with in-context conditioning."""
    layer = dit.DiTConditioningBlock(
        features=16,
        num_heads=4,
        ffn_ratio=4,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 8, 16), dtype=dtype)
    test_cond = jnp.ones((2, 16), dtype=dtype)
    variables = layer.init(
        jax.random.PRNGKey(0),
        test_input,
        cond=test_cond,
        deterministic=True,
    )
    test_output = layer.apply(
        variables,
        test_input,
        cond=test_cond,
        deterministic=False,
        rngs={"dropout": jax.random.PRNGKey(1)},
    )
    assert isinstance(test_output, jax.Array)
    assert test_output.shape == (2, 9, 16)
    assert test_output.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_cross_attention_dit_block(dtype: typing.Any) -> None:
    r"""Test the DiT block with cross-attention."""
    layer = dit.DiTCrossAttentionBlock(
        features=16,
        num_heads=4,
        ffn_ratio=4,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 8, 16), dtype=dtype)
    test_cond = jnp.ones((2, 16), dtype=dtype)
    variables = layer.init(
        jax.random.PRNGKey(0),
        test_input,
        cond=test_cond,
        deterministic=True,
    )
    test_output = layer.apply(
        variables,
        test_input,
        cond=test_cond,
        deterministic=False,
        rngs={"dropout": jax.random.PRNGKey(1)},
    )
    assert isinstance(test_output, jax.Array)
    assert test_output.shape == (2, 8, 16)
    assert test_output.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_adaln_dit_block(dtype: typing.Any) -> None:
    r"""Test the DiT block with adaptive layer normalization."""
    layer = dit.DiTAdaLNBlock(
        features=16,
        num_heads=4,
        ffn_ratio=4,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 8, 16), dtype=dtype)
    test_cond = jnp.ones((2, 16), dtype=dtype)
    variables = layer.init(
        jax.random.PRNGKey(0),
        test_input,
        cond=test_cond,
        deterministic=True,
    )
    test_output = layer.apply(
        variables,
        test_input,
        cond=test_cond,
        deterministic=False,
        rngs={"dropout": jax.random.PRNGKey(1)},
    )
    assert isinstance(test_output, jax.Array)
    assert test_output.shape == (2, 8, 16)
    assert test_output.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_standard_decoder(dtype: typing.Any) -> None:
    r"""Test the standard DiT decoder."""
    layer = dit.StandardDecoder(
        features=3,
        patch_size=2,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 8, 16), dtype=dtype)
    test_cond = jnp.ones((2, 16), dtype=dtype)
    variables = layer.init(jax.random.PRNGKey(0), test_input, cond=test_cond)
    test_output = layer.apply(
        variables,
        test_input,
        cond=test_cond,
        rngs={"dropout": jax.random.PRNGKey(1)},
    )
    assert isinstance(test_output, jax.Array)
    assert test_output.shape == (2, 8, 12)
    assert test_output.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_adaln_decoder(dtype: typing.Any) -> None:
    r"""Test the adaptive layer norm DiT decoder."""
    layer = dit.AdaLNDecoder(
        features=3,
        patch_size=2,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 8, 16), dtype=dtype)
    test_cond = jnp.ones((2, 16), dtype=dtype)
    variables = layer.init(jax.random.PRNGKey(0), test_input, cond=test_cond)
    test_output = layer.apply(
        variables,
        test_input,
        cond=test_cond,
        rngs={"dropout": jax.random.PRNGKey(1)},
    )
    assert isinstance(test_output, jax.Array)
    assert test_output.shape == (2, 8, 12)
    assert test_output.dtype == dtype


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
