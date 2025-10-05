import functools
import sys
import typing

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
import pytest

from learning.generative.model import refinenet


@pytest.mark.parametrize("out_channels", [1, 3])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_conv_1x1(out_channels: int, dtype: typing.Any) -> None:
    """Test 1x1 convolution builder."""
    layer = refinenet._conv_1x1(
        out_channels=out_channels,
        name="conv1",
        dtype=dtype,
        param_dtype=dtype,
    )
    assert isinstance(layer, nn.Conv)
    assert layer.features == out_channels
    assert layer.kernel_size == (1, 1)
    assert layer.strides == (1, 1)
    assert layer.padding == (0, 0)
    assert layer.name == "conv1"
    variables = layer.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))
    chex.assert_shape(variables["params"]["kernel"], (1, 1, 3, out_channels))
    chex.assert_type(variables["params"]["kernel"], dtype)
    chex.assert_shape(variables["params"]["bias"], (out_channels,))
    chex.assert_type(variables["params"]["bias"], dtype)
    test_output = layer.apply(variables, jnp.ones((1, 32, 32, 3)))
    chex.assert_type(test_output, dtype)
    chex.assert_shape(test_output, (1, 32, 32, out_channels))


@pytest.mark.parametrize("out_channels", [1, 3])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_conv_3x3(out_channels: int, dtype: typing.Any) -> None:
    """Test 3x3 convolution builder."""
    layer = refinenet._conv_3x3(
        out_channels=out_channels,
        name="conv3",
        dtype=dtype,
        param_dtype=dtype,
    )
    assert isinstance(layer, nn.Conv)
    assert layer.features == out_channels
    assert layer.kernel_size == (3, 3)
    assert layer.strides == (1, 1)
    assert layer.padding == (1, 1)
    assert layer.name == "conv3"
    variables = layer.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))
    chex.assert_shape(variables["params"]["kernel"], (3, 3, 3, out_channels))
    chex.assert_type(variables["params"]["kernel"], dtype)
    chex.assert_shape(variables["params"]["bias"], (out_channels,))
    chex.assert_type(variables["params"]["bias"], dtype)
    test_output = layer.apply(variables, jnp.ones((1, 32, 32, 3)))
    chex.assert_type(test_output, dtype)
    chex.assert_shape(test_output, (1, 32, 32, out_channels))


@pytest.mark.parametrize("out_channels", [1, 3])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_dilated_conv_3x3(
    out_channels: int,
    dilation: int,
    dtype: typing.Any,
) -> None:
    """Test dilated 3x3 convolution builder."""
    layer = refinenet._dilated_conv_3x3(
        out_channels=out_channels,
        dilation=dilation,
        name="dilated_conv3",
        dtype=dtype,
        param_dtype=dtype,
    )
    assert isinstance(layer, nn.Conv)
    assert layer.features == out_channels
    assert layer.kernel_size == (3, 3)
    assert layer.strides == (1, 1)
    assert layer.kernel_dilation == (dilation, dilation)
    assert layer.padding == (dilation, dilation)
    assert layer.name == "dilated_conv3"
    variables = layer.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))
    chex.assert_shape(variables["params"]["kernel"], (3, 3, 3, out_channels))
    chex.assert_type(variables["params"]["kernel"], dtype)
    chex.assert_shape(variables["params"]["bias"], (out_channels,))
    chex.assert_type(variables["params"]["bias"], dtype)
    test_output = layer.apply(variables, jnp.ones((1, 32, 32, 3)))
    chex.assert_type(test_output, dtype)
    chex.assert_shape(test_output, (1, 32, 32, out_channels))


@pytest.mark.parametrize("features", [1, 3])
@pytest.mark.parametrize("num_classes", [5, 10])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_conditional_instance_norm_2d_plus(
    features: int,
    num_classes: int,
    use_bias: bool,
    dtype: typing.Any,
) -> None:
    """Test `ConditionalInstanceNorm2dPlus` layer."""
    layer = refinenet.ConditionalInstanceNorm2dPlus(
        features=features,
        num_classes=num_classes,
        use_bias=use_bias,
        dtype=dtype,
        param_dtype=dtype,
    )
    assert isinstance(layer, nn.Module)
    variables = layer.init(
        jax.random.PRNGKey(0),
        jnp.ones((1, 32, 32, features), dtype=dtype),
        jnp.ones((1,), dtype=jnp.int32),
    )
    # in this layer, instance norm does not have affine params
    assert variables["params"].get("instance_norm") is None

    if use_bias:
        chex.assert_shape(
            variables["params"]["embed"]["embedding"],
            (num_classes, 3 * features),
        )
    else:
        chex.assert_shape(
            variables["params"]["embed"]["embedding"],
            (num_classes, 2 * features),
        )
    chex.assert_type(variables["params"]["embed"]["embedding"], dtype)
    test_output = layer.apply(
        variables,
        jnp.ones((1, 32, 32, features)),
        jnp.ones((1,), dtype=jnp.int32),
    )
    chex.assert_type(test_output, dtype)
    chex.assert_shape(test_output, (1, 32, 32, features))


@pytest.mark.parametrize("features", [1, 3])
@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_conv_mean_pool(
    features: int,
    kernel_size: int,
    dtype: typing.Any,
) -> None:
    """Test `ConvMeanPool` layer."""
    layer = refinenet.ConvMeanPool(
        features=features,
        kernel_size=kernel_size,
        name="conv_mean_pool",
        dtype=dtype,
        param_dtype=dtype,
    )
    assert isinstance(layer, nn.Module)
    assert layer.features == features
    assert layer.kernel_size == kernel_size
    assert layer.name == "conv_mean_pool"
    variables = layer.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))
    chex.assert_shape(
        variables["params"]["conv"]["kernel"],
        (kernel_size, kernel_size, 3, features),
    )
    chex.assert_type(variables["params"]["conv"]["kernel"], dtype)
    chex.assert_shape(variables["params"]["conv"]["bias"], (features,))
    chex.assert_type(variables["params"]["conv"]["bias"], dtype)
    test_output = layer.apply(variables, jnp.ones((1, 32, 32, 3)))
    chex.assert_type(test_output, dtype)
    chex.assert_shape(test_output, (1, 16, 16, features))


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("out_channels", [1, 3])
@pytest.mark.parametrize("dilation", [None, 1, 2])
@pytest.mark.parametrize("resample", [None, "up", "down"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_conditional_residual_block(
    in_channels: int,
    out_channels: int,
    dilation: typing.Optional[int],
    resample: typing.Optional[str],
    dtype: typing.Any,
) -> None:
    """Test `ConditionalResidualBlock` module."""
    if resample not in (None, "down"):
        with pytest.raises(ValueError):
            block = refinenet.ConditionalResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_module=functools.partial(
                    refinenet.ConditionalInstanceNorm2dPlus,
                    num_classes=10,
                ),
                dilation=dilation,
                resample=resample,
                dtype=dtype,
                param_dtype=dtype,
            )
            _ = block.init(
                jax.random.PRNGKey(0),
                jnp.ones((2, 32, 32, in_channels), dtype=dtype),
                jnp.ones((2,), dtype=jnp.int32),
            )
        return

    block = refinenet.ConditionalResidualBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        norm_module=functools.partial(
            refinenet.ConditionalInstanceNorm2dPlus,
            num_classes=10,
        ),
        dilation=dilation,
        resample=resample,
        dtype=dtype,
        param_dtype=dtype,
        name="res_block",
    )
    assert isinstance(block, nn.Module)
    assert block.in_channels == in_channels
    assert block.out_channels == out_channels
    assert block.dilation == dilation
    assert block.resample == resample
    assert block.name == "res_block"
    variables = block.init(
        jax.random.PRNGKey(0),
        jnp.ones((2, 32, 32, in_channels), dtype=dtype),
        jnp.ones((2,), dtype=jnp.int32),
    )
    if resample == "down":
        if dilation is not None:
            chex.assert_shape(
                variables["params"]["conv1"]["kernel"],
                (3, 3, in_channels, in_channels),
            )
            chex.assert_shape(
                variables["params"]["conv1"]["bias"],
                (in_channels,),
            )
            chex.assert_shape(
                variables["params"]["conv2"]["kernel"],
                (3, 3, in_channels, out_channels),
            )
            chex.assert_shape(
                variables["params"]["conv2"]["bias"],
                (out_channels,),
            )
            if in_channels != out_channels:
                chex.assert_shape(
                    variables["params"]["shortcut"]["kernel"],
                    (3, 3, in_channels, out_channels),
                )
                chex.assert_shape(
                    variables["params"]["shortcut"]["bias"],
                    (out_channels,),
                )
        else:
            chex.assert_shape(
                variables["params"]["conv1"]["kernel"],
                (3, 3, in_channels, in_channels),
            )
            chex.assert_shape(
                variables["params"]["conv1"]["bias"],
                (in_channels,),
            )
            chex.assert_shape(
                variables["params"]["conv2"]["conv"]["kernel"],
                (3, 3, in_channels, out_channels),
            )
            chex.assert_shape(
                variables["params"]["conv2"]["conv"]["bias"],
                (out_channels,),
            )
            if in_channels != out_channels:
                chex.assert_shape(
                    variables["params"]["shortcut"]["conv"]["kernel"],
                    (1, 1, in_channels, out_channels),
                )
                chex.assert_shape(
                    variables["params"]["shortcut"]["conv"]["bias"],
                    (out_channels,),
                )
    else:
        # test case: resample is None
        if dilation is not None:
            chex.assert_shape(
                variables["params"]["conv1"]["kernel"],
                (3, 3, in_channels, out_channels),
            )
            chex.assert_shape(
                variables["params"]["conv1"]["bias"],
                (out_channels,),
            )
            chex.assert_shape(
                variables["params"]["conv2"]["kernel"],
                (3, 3, out_channels, out_channels),
            )
            chex.assert_shape(
                variables["params"]["conv2"]["bias"],
                (out_channels,),
            )
            if in_channels != out_channels:
                chex.assert_shape(
                    variables["params"]["shortcut"]["kernel"],
                    (3, 3, in_channels, out_channels),
                )
                chex.assert_shape(
                    variables["params"]["shortcut"]["bias"],
                    (out_channels,),
                )
        else:
            chex.assert_shape(
                variables["params"]["conv1"]["kernel"],
                (3, 3, in_channels, out_channels),
            )
            chex.assert_shape(
                variables["params"]["conv1"]["bias"],
                (out_channels,),
            )
            chex.assert_shape(
                variables["params"]["conv2"]["kernel"],
                (3, 3, out_channels, out_channels),
            )
            chex.assert_shape(
                variables["params"]["conv2"]["bias"],
                (out_channels,),
            )
            if in_channels != out_channels:
                chex.assert_shape(
                    variables["params"]["shortcut"]["kernel"],
                    (1, 1, in_channels, out_channels),
                )
                chex.assert_shape(
                    variables["params"]["shortcut"]["bias"],
                    (out_channels,),
                )

    test_output = block.apply(
        variables,
        jnp.ones((2, 32, 32, in_channels), dtype=dtype),
        jnp.ones((2,), dtype=jnp.int32),
    )
    chex.assert_type(test_output, dtype)
    if resample == "down" and dilation is None:
        chex.assert_shape(test_output, (2, 16, 16, out_channels))
    else:
        chex.assert_shape(test_output, (2, 32, 32, out_channels))


@pytest.mark.parametrize("features", [1, 3])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_conditional_rcu_block(features: int, dtype: typing.Any) -> None:
    """Test the `ConditionalRCUBlock` module."""
    block = refinenet.ConditionalRCUBlock(
        features=features,
        norm_module=functools.partial(
            refinenet.ConditionalInstanceNorm2dPlus,
            num_classes=10,
        ),
        num_blocks=2,
        num_stages=2,
        dtype=dtype,
        param_dtype=dtype,
    )
    assert isinstance(block, nn.Module)
    variables = block.init(
        jax.random.PRNGKey(0),
        jnp.ones((1, 32, 32, features), dtype=dtype),
        jnp.ones((1,), dtype=jnp.int32),
    )
    for i in range(2):
        for j in range(2):
            chex.assert_shape(
                variables["params"][f"{i+1}_{j+1}_conv"]["kernel"],
                (3, 3, features, features),
            )
            chex.assert_type(
                variables["params"][f"{i+1}_{j+1}_conv"]["kernel"],
                dtype,
            )
            assert variables["params"][f"{i+1}_{j+1}_conv"].get("bias") is None

    test_output = block.apply(
        variables,
        jnp.ones((1, 32, 32, features), dtype=dtype),
        jnp.ones((1,), dtype=jnp.int32),
    )
    chex.assert_type(test_output, dtype)
    chex.assert_shape(test_output, (1, 32, 32, features))


@pytest.mark.parametrize("features", [1, 3])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_conditional_msf_block(features: int, dtype: typing.Any) -> None:
    """Test the `ConditionalMSFBlock` module."""
    block = refinenet.ConditionalMSFBlock(
        in_features=(3, 8),
        features=features,
        norm_module=functools.partial(
            refinenet.ConditionalInstanceNorm2dPlus,
            num_classes=10,
        ),
        dtype=dtype,
        param_dtype=dtype,
    )
    assert isinstance(block, nn.Module)
    with pytest.raises(AssertionError):
        _ = block.init(
            jax.random.PRNGKey(0),
            inputs=[
                jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
            ],
            cond=jnp.ones((2,), dtype=jnp.int32),
            shape=(28, 28),
        )

    variables = block.init(
        jax.random.PRNGKey(0),
        inputs=[
            jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
            jnp.ones((2, 16, 16, 8), dtype=jnp.float32),
        ],
        cond=jnp.ones((2,), dtype=jnp.int32),
        shape=(28, 28),
    )
    for i in range(2):
        chex.assert_shape(
            variables["params"][f"convs.{i:d}"]["kernel"],
            (3, 3, (3 if i == 0 else 8), features),
        )
        chex.assert_type(
            variables["params"][f"convs.{i:d}"]["kernel"],
            dtype,
        )
        chex.assert_shape(
            variables["params"][f"convs.{i:d}"]["bias"],
            (features,),
        )
        chex.assert_type(
            variables["params"][f"convs.{i:d}"]["bias"],
            dtype,
        )

    test_output = block.apply(
        variables,
        inputs=[
            jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
            jnp.ones((2, 16, 16, 8), dtype=jnp.float32),
        ],
        cond=jnp.ones((2,), dtype=jnp.int32),
        shape=(28, 28),
    )
    chex.assert_type(test_output, jnp.float32)
    chex.assert_shape(test_output, (2, 28, 28, features))


@pytest.mark.parametrize("features", [1, 3])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_conditional_crp_block(features: int, dtype: typing.Any) -> None:
    """Test the `ConditionalCRPBlock` module."""
    block = refinenet.ConditionalCRPBlock(
        features=features,
        norm_module=functools.partial(
            refinenet.ConditionalInstanceNorm2dPlus,
            num_classes=10,
        ),
        num_stages=2,
        dtype=dtype,
        param_dtype=dtype,
    )
    assert isinstance(block, nn.Module)
    variables = block.init(
        jax.random.PRNGKey(0),
        jnp.ones((1, 32, 32, features), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.int32),
    )
    for i in range(2):
        chex.assert_shape(
            variables["params"][f"convs.{i:d}"]["kernel"],
            (3, 3, features, features),
        )
        chex.assert_type(
            variables["params"][f"convs.{i:d}"]["kernel"],
            jnp.float32,
        )
        assert variables["params"][f"convs.{i:d}"].get("bias") is None

    test_output = block.apply(
        variables,
        jnp.ones((1, 32, 32, features), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.int32),
    )
    chex.assert_type(test_output, jnp.float32)
    chex.assert_shape(test_output, (1, 32, 32, features))


@pytest.mark.parametrize("features", [1, 3])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("is_last_block", [True, False])
def test_conditional_refine_block(
    features: int,
    dtype: typing.Any,
    is_last_block: bool,
) -> None:
    """Test the `ConditionalRefineBlock` module."""
    test_inputs = [
        jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
        jnp.ones((2, 16, 16, 8), dtype=jnp.float32),
    ]
    block = refinenet.ConditionalRefineBlock(
        in_features=[3, 8],
        out_features=features,
        norm_module=functools.partial(
            refinenet.ConditionalInstanceNorm2dPlus,
            num_classes=10,
        ),
        is_last_block=is_last_block,
        name="refine_block",
        dtype=dtype,
        param_dtype=dtype,
    )
    assert isinstance(block, nn.Module)
    assert block.in_features == [3, 8]
    assert block.out_features == features
    assert block.name == "refine_block"
    with pytest.raises(AssertionError):
        _ = block.init(
            jax.random.PRNGKey(0),
            inputs=[
                jnp.ones((2, 32, 32, 3), dtype=jnp.float32),
            ],
            cond=jnp.ones((2,), dtype=jnp.int32),
            output_shape=(28, 28),
        )
    variables = block.init(
        jax.random.PRNGKey(0),
        inputs=test_inputs,
        cond=jnp.ones((2,), dtype=jnp.int32),
        output_shape=(28, 28),
    )
    test_output = block.apply(
        variables,
        inputs=test_inputs,
        cond=jnp.ones((2,), dtype=jnp.int32),
        output_shape=(28, 28),
    )
    chex.assert_type(test_output, jnp.float32)
    chex.assert_shape(test_output, (2, 28, 28, features))


def test_conditional_refinenet() -> None:
    """Integrated test for the `ConditionalRefineNet` module."""
    model = refinenet.ConditionalRefineNet(
        in_channels=3,
        image_size=32,
        latent_channels=16,
        norm_module=functools.partial(
            refinenet.ConditionalInstanceNorm2dPlus,
            num_classes=10,
        ),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    assert isinstance(model, nn.Module)
    with pytest.raises(AssertionError):
        _ = model.init(
            jax.random.PRNGKey(0),
            jnp.ones((2, 28, 28, 1), dtype=jnp.float32),
            jnp.ones((2,), dtype=jnp.int32),
        )
    test_input = jnp.ones((2, 32, 32, 3), dtype=jnp.float32)
    variables = model.init(
        jax.random.PRNGKey(0),
        test_input,
        jnp.ones((2,), dtype=jnp.int32),
    )
    test_output = model.apply(
        variables,
        test_input,
        jnp.ones((2,), dtype=jnp.int32),
    )
    chex.assert_type(test_output, jnp.float32)
    chex.assert_shape(test_output, (2, 32, 32, 3))


if __name__ == "__main__":
    sys.exit(pytest.main(["-xv", __file__]))
