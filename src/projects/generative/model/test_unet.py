import sys
import typing

import jax
from jax import numpy as jnp
import pytest

from src.projects.generative.model import unet


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_down_block(dtype: typing.Any) -> None:
    r"""Tests the residual downsampling block in U-Net models."""
    rng = jax.random.PRNGKey(42)

    block = unet.DownResNetBlock(features=64, dtype=dtype, param_dtype=dtype)
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


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
