import typing

from flax import linen as nn
from flax import typing as flax_typing
import jax
from jax import numpy as jnp


# ==============================================================================
# Components
# ==============================================================================
class ConvBNReLU(nn.Module):
    r"""A convolution-batchnorm-relu block in Inception architecture.

    Args:
        features (int): Dimensionality of the output feature map.
        kernel_size (int): Size of the 2D convolution kernel.
        strides (int): Stride of the 2D convolution.
        padding (PaddingLike): Padding for the 2D convolution.
            Default is `"VALID"`.
        use_bias (bool): Whether to use bias in the convolution.
            Default is `False`.
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
    """

    features: int
    kernel_size: int
    strides: int
    padding: flax_typing.PaddingLike = "VALID"
    use_bias: bool = False
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass of the Conv-BN-ReLU block.

        Args:
            inputs (jax.Array): Input array of shape `(*, height, width, C)`.
            deterministic (bool, optional): Whether to apply running averages
                in batch normalization.

        Returns:
            Output array of shape `(*, new_height, new_width, features)`.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )

        conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.strides, self.strides),
            padding=self.padding,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv",
        )
        out = conv(inputs)
        bn = nn.BatchNorm(
            use_running_average=m_deterministic,
            epsilon=0.001,
            momentum=0.1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="bn",
        )
        out = bn(out)
        out = jax.nn.relu(out)

        return out


class InceptionABlock(nn.Module):
    r"""An Inception block comprises 5x5, 3x3, 1x1 convolutions and avg pooling.

    Args:
        pooled_features (int): Number of output features for the pooling branch.
        deterministic (bool, optional): Whether to apply running averages
            in batch normalization.
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
    """

    pooled_features: int
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass of the Inception-A block.

        Args:
            inputs (jax.Array): Input array of shape `(*, height, width, C)`.
            deterministic (bool, optional): Whether to apply running averages
                in batch normalization.

        Returns:
            Output array of shape `(*, new_height, new_width, features)`.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )

        # branch 1: 1x1 convolution
        branch_1x1_conv = ConvBNReLU(
            features=64,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch_1x1",
        )
        out_1x1 = branch_1x1_conv(inputs, deterministic=m_deterministic)

        # branch 2: 1x1 convolution followed by 5x5 convolution
        branch_5x5_1 = ConvBNReLU(
            features=48,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch_5x5_1",
        )
        branch_5x5_2 = ConvBNReLU(
            features=64,
            kernel_size=5,
            strides=1,
            padding=2,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch_5x5_2",
        )
        out_5x5 = branch_5x5_1(inputs, deterministic=m_deterministic)
        out_5x5 = branch_5x5_2(out_5x5, deterministic=m_deterministic)

        # branch 3: 1x1 convolution followed by two 3x3 convolutions
        branch_3x3_1 = ConvBNReLU(
            features=64,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_1",
        )
        branch_3x3_2 = ConvBNReLU(
            features=96,
            kernel_size=3,
            strides=1,
            padding=1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_2",
        )
        branch_3x3_3 = ConvBNReLU(
            features=96,
            kernel_size=3,
            strides=1,
            padding=1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_3",
        )
        out_3x3 = branch_3x3_1(inputs, deterministic=m_deterministic)
        out_3x3 = branch_3x3_2(out_3x3, deterministic=m_deterministic)
        out_3x3 = branch_3x3_3(out_3x3, deterministic=m_deterministic)

        # branch 4: average pooling followed by 1x1 convolution
        branch_pool = ConvBNReLU(
            features=self.pooled_features,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch_pool",
        )
        out_pool = nn.avg_pool(
            inputs,
            window_shape=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )
        out_pool = branch_pool(out_pool, deterministic=m_deterministic)

        return jnp.concatenate(
            [out_1x1, out_5x5, out_3x3, out_pool],
            axis=-1,
        )


class InceptionBBlock(nn.Module):
    r"""An Inception block comprises 3x3, 1x1 convolution and max pooling.

    Args:
        deterministic (bool, optional): Whether to apply running averages
            in batch normalization.
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
    """

    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(
        self, inputs: jax.Array, deterministic: typing.Optional[bool] = None
    ) -> jax.Array:
        r"""Forward pass of the Inception-B block.

        Args:
            inputs (jax.Array): Input array of shape `(*, height, width, C)`.
            deterministic (bool, optional): Whether to apply running averages
                in batch normalization.

        Returns:
            Output array of shape `(*, new_height, new_width, features)`.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )

        # branch 1: 3x3 convolution with stride 2
        branch_3x3 = ConvBNReLU(
            features=384,
            kernel_size=3,
            strides=2,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch_3x3",
        )
        out_3x3 = branch_3x3(inputs, deterministic=m_deterministic)

        # branch 2: a cascade of three convolutions
        branch_3x3dbl_1 = ConvBNReLU(
            features=64,
            kernel_size=1,
            strides=1,
            padding=0,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch_3x3dbl_1",
        )
        branch_3x3dbl_2 = ConvBNReLU(
            features=96,
            kernel_size=3,
            strides=1,
            padding=1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch_3x3dbl_2",
        )
        branch_3x3dbl_3 = ConvBNReLU(
            features=96,
            kernel_size=3,
            strides=2,
            padding=0,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch_3x3dbl_3",
        )
        out_3x3dbl = branch_3x3dbl_1(inputs, deterministic=m_deterministic)
        out_3x3dbl = branch_3x3dbl_2(out_3x3dbl, deterministic=m_deterministic)
        out_3x3dbl = branch_3x3dbl_3(out_3x3dbl, deterministic=m_deterministic)

        # branch 3: max pooling
        out_pool = nn.max_pool(
            inputs,
            window_shape=(3, 3),
            strides=(2, 2),
            padding="VALID",
        )

        return jnp.concatenate(
            [out_3x3, out_3x3dbl, out_pool],
            axis=-1,
        )
