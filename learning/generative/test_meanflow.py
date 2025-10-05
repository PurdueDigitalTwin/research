import sys
import typing

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
import pytest

from learning.generative import meanflow


@pytest.mark.parametrize("distribution", ["uniform", "normal", "lognormal"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_sample_t_r(distribution: str, dtype: typing.Any) -> None:
    """Test the `sample_t_r` function."""
    key = jax.random.PRNGKey(0)
    shape = (2, 3)

    if distribution not in ["uniform", "lognormal"]:
        with pytest.raises(ValueError):
            meanflow.sample_t_r(
                key=key,
                shape=shape,
                dtype=dtype,
                distribution=distribution,
            )
        return

    # Test uniform distribution
    t, r = meanflow.sample_t_r(
        key=key,
        shape=shape,
        dtype=dtype,
        distribution="uniform",
    )
    chex.assert_shape(t, shape)
    chex.assert_shape(r, shape)
    chex.assert_type(t, dtype)
    chex.assert_type(r, dtype)
    chex.assert_tree_all_finite(t)
    chex.assert_tree_all_finite(r)
    assert jnp.all(t >= 0) and jnp.all(t <= 1)
    assert jnp.all(r >= 0) and jnp.all(r <= 1)


@pytest.mark.parametrize("features", [1, 8])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_timestamp_embed(features: int, dtype: typing.Any) -> None:
    """Test the `TimestampEmbed` module."""
    embed = meanflow.TimestampEmbed(
        features=features,
        frequency=256,
        name="timestamp_embed",
        dtype=dtype,
        param_dtype=dtype,
    )
    assert isinstance(embed, nn.Module)
    assert embed.features == features
    assert embed.frequency == 256
    assert embed.dtype == dtype
    assert embed.param_dtype == dtype
    variables = embed.init(
        jax.random.PRNGKey(0),
        jnp.ones((2,), dtype=jnp.int32),
    )
    chex.assert_shape(variables["params"]["fc_in"]["kernel"], (256, features))
    chex.assert_type(variables["params"]["fc_in"]["kernel"], dtype)
    chex.assert_shape(variables["params"]["fc_in"]["bias"], (features,))
    chex.assert_type(variables["params"]["fc_in"]["bias"], dtype)
    chex.assert_shape(
        variables["params"]["fc_out"]["kernel"],
        (features, features),
    )
    chex.assert_type(variables["params"]["fc_out"]["kernel"], dtype)
    chex.assert_shape(variables["params"]["fc_out"]["bias"], (features,))
    chex.assert_type(variables["params"]["fc_out"]["bias"], dtype)

    test_output = embed.apply(
        variables,
        jnp.array([10, 1000], dtype=jnp.int32),
    )
    chex.assert_shape(test_output, (2, features))
    chex.assert_type(test_output, dtype)
    chex.assert_tree_all_finite(test_output)


@pytest.mark.parametrize("features", [1, 8])
@pytest.mark.parametrize("use_cfg_embedding", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_condition_embed(
    features: int,
    use_cfg_embedding: bool,
    dtype: typing.Any,
) -> None:
    """Test the `ConditionEmbed` module."""
    if use_cfg_embedding:
        # TODO: implement classifier-free guidance.
        pytest.skip("Classifier-free guidance not supported yet.")

    embed = meanflow.ConditionEmbed(
        features=features,
        num_classes=10,
        use_cfg_embedding=use_cfg_embedding,
        name="condition_embed",
        dtype=dtype,
        param_dtype=dtype,
    )
    assert isinstance(embed, nn.Module)
    assert embed.features == features
    assert embed.num_classes == 10
    assert embed.use_cfg_embedding == use_cfg_embedding
    assert embed.dtype == dtype
    assert embed.param_dtype == dtype
    variables = embed.init(
        jax.random.PRNGKey(0),
        jnp.ones((2,), dtype=jnp.int32),
    )
    chex.assert_shape(
        variables["params"]["embedding_table"]["embedding"],
        (10 + int(use_cfg_embedding), features),
    )
    chex.assert_type(
        variables["params"]["embedding_table"]["embedding"], dtype
    )

    test_output = embed.apply(
        variables,
        jnp.array([1, 9], dtype=jnp.int32),
    )
    chex.assert_shape(test_output, (2, features))
    chex.assert_type(test_output, dtype)
    chex.assert_tree_all_finite(test_output)


@pytest.mark.parametrize("features", [1, 8])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_conditional_instance_norm(
    features: int,
    use_bias: bool,
    dtype: typing.Any,
) -> None:
    """Test the `ConditionalInstanceNorm` module."""
    cond_features = 4
    cond = jnp.ones((2, cond_features), dtype=dtype)

    norm = meanflow.ConditionalInstanceNorm(
        features=features,
        use_bias=use_bias,
        name="conditional_instance_norm",
        dtype=dtype,
        param_dtype=dtype,
    )
    assert isinstance(norm, nn.Module)
    assert norm.features == features
    assert norm.use_bias == use_bias
    assert norm.dtype == dtype
    assert norm.param_dtype == dtype
    variables = norm.init(
        jax.random.PRNGKey(0),
        jnp.ones((2, 16, 16, features), dtype=dtype),
        cond,
    )
    assert variables["params"].get("instance_norm") is None
    if use_bias:
        chex.assert_shape(
            variables["params"]["embed"]["kernel"],
            (cond_features, features * 3),
        )
        assert variables["params"]["embed"].get("bias") is None
    else:
        chex.assert_shape(
            variables["params"]["embed"]["kernel"],
            (cond_features, features * 2),
        )
        assert variables["params"]["embed"].get("bias") is None

    test_output = norm.apply(
        variables,
        jnp.ones((2, 16, 16, features), dtype=dtype),
        cond,
    )
    chex.assert_shape(test_output, (2, 16, 16, features))
    chex.assert_type(test_output, dtype)
    chex.assert_tree_all_finite(test_output)


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
