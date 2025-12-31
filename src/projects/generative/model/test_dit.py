import sys
import typing

import jax
from jax import numpy as jnp
import pytest

from src.projects.generative.model import dit


def test_sinusoidal_pos_enc() -> None:
    r"""Test the sinusoidal positional encoding function."""
    test_pos = jnp.array([0, 1, 2], dtype=jnp.float32)
    test_features = 6
    test_output = dit.sinusoidal_pos_enc(test_features, test_pos)
    assert isinstance(test_output, jax.Array)
    assert test_output.shape == (3, test_features)
    assert test_output.dtype == jnp.float32


@pytest.mark.parametrize("num_extra_tokens", [0, 2])
def test_sinusoidal_patch_enc(num_extra_tokens: int) -> None:
    r"""Test the sinusoidal patch positional encoding function."""
    test_output = dit.sinusoidal_patch_enc(
        features=8,
        grid_size=16,
        num_extra_tokens=num_extra_tokens,
    )
    assert isinstance(test_output, jax.Array)
    assert test_output.shape == (16 * 16 + num_extra_tokens, 8)
    assert test_output.dtype == jnp.float32


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


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_patch_embed(dtype: typing.Any) -> None:
    r"""Test the DiT patch embedding module."""
    layer = dit.PatchEmbed(
        features=64,
        patch_size=2,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 8, 8, 3), dtype=dtype)
    variables = layer.init(jax.random.PRNGKey(0), test_input)
    test_output = layer.apply(
        variables,
        test_input,
        rngs={"dropout": jax.random.PRNGKey(1)},
    )
    assert isinstance(test_output, jax.Array)
    assert test_output.shape == (2, 16, 64)
    assert test_output.dtype == dtype


@pytest.mark.parametrize(
    "block_type",
    ["adaLN", "cross_attention", "in_context"],
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_dit(block_type: str, dtype: typing.Any) -> None:
    r"""Test the DiT generative model."""
    layer = dit.DiffusionTransformer(
        features=16,
        depth=2,
        num_heads=4,
        ffn_ratio=4,
        patch_size=4,
        block_type="adaLN",
        learn_sigma=True,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 32, 32, 4), dtype=dtype)
    test_timestamp = jnp.ones((2,), dtype=dtype)
    test_label = jnp.ones((2,), dtype=jnp.int32)
    variables = layer.init(
        jax.random.PRNGKey(0),
        test_input,
        timestamp=test_timestamp,
        labels=test_label,
        deterministic=True,
    )
    test_output = layer.apply(
        variables,
        test_input,
        timestamp=test_timestamp,
        labels=test_label,
        deterministic=False,
        rngs={"dropout": jax.random.PRNGKey(1)},
    )
    assert isinstance(test_output, jax.Array)
    assert test_output.shape == (2, 32, 32, 8)
    assert test_output.dtype == dtype


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
