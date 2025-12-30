import sys
import typing

import jax
from jax import numpy as jnp
import pytest

from src.projects.generative.model import dit


@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_self_attention(num_heads: int, dtype: typing.Any) -> None:
    r"""Test the multi-head self-attention module for Diffusion Transformer."""
    layer = dit.SelfAttention(
        features=16,
        num_heads=num_heads,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((2, 4, 16), dtype=dtype)
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


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
