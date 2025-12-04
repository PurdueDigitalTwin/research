import sys
import typing

import jax
from jax import numpy as jnp
import pytest

from src.projects.generative.tools.fid import inception


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_conv_bn_relu(dtype: typing.Any) -> None:
    r"""Test the `ConvBNReLU` module."""
    module = inception.ConvBNReLU(
        features=32,
        kernel_size=3,
        strides=2,
        padding="VALID",
        use_bias=False,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((1, 299, 299, 3), dtype=dtype)
    variables = module.init(
        jax.random.PRNGKey(0),
        test_input,
        deterministic=True,
    )
    output = module.apply(variables, test_input, deterministic=True)
    assert isinstance(output, jax.Array)
    assert output.shape == (1, 149, 149, 32)
    assert output.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_inception_a_block(dtype: typing.Any) -> None:
    r"""Test the `InceptionABlock` module."""
    module = inception.InceptionABlock(
        pooled_features=32,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((1, 35, 35, 192), dtype=dtype)
    variables = module.init(
        jax.random.PRNGKey(0),
        test_input,
        deterministic=True,
    )
    output = module.apply(variables, test_input, deterministic=True)
    assert isinstance(output, jax.Array)
    assert output.shape == (1, 35, 35, 256)
    assert output.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_inception_b_block(dtype: typing.Any) -> None:
    r"""Test the `InceptionBBlock` module."""
    module = inception.InceptionBBlock(
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((1, 35, 35, 288), dtype=dtype)
    variables = module.init(
        jax.random.PRNGKey(0),
        test_input,
        deterministic=True,
    )
    output = module.apply(variables, test_input, deterministic=True)
    assert isinstance(output, jax.Array)
    assert output.shape == (1, 17, 17, 768)
    assert output.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_inception_c_block(dtype: typing.Any) -> None:
    r"""Test the `InceptionCBlock` module."""
    module = inception.InceptionCBlock(
        features=128,
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((1, 17, 17, 768), dtype=dtype)
    variables = module.init(
        jax.random.PRNGKey(0),
        test_input,
        deterministic=True,
    )
    output = module.apply(variables, test_input, deterministic=True)
    assert isinstance(output, jax.Array)
    assert output.shape == (1, 17, 17, 768)
    assert output.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_inception_d_block(dtype: typing.Any) -> None:
    r"""Test the `InceptionDBlock` module."""
    module = inception.InceptionDBlock(
        dtype=dtype,
        param_dtype=dtype,
    )
    test_input = jnp.ones((1, 17, 17, 768), dtype=dtype)
    variables = module.init(
        jax.random.PRNGKey(0),
        test_input,
        deterministic=True,
    )
    output = module.apply(variables, test_input, deterministic=True)
    assert isinstance(output, jax.Array)
    assert output.shape == (1, 8, 8, 1280)
    assert output.dtype == dtype


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
