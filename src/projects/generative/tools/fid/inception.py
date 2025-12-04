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
        kernel_size (int or tuple): Size of the convolutional kernel.
        strides (int or tuple): Stride of the convolution.
        padding (PaddingLike): Padding for the 2D convolution.
            Default is `"VALID"`.
        use_bias (bool): Whether to use bias in the convolution.
            Default is `False`.
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
    """

    features: int
    kernel_size: typing.Union[int, typing.Tuple[int, int]]
    strides: typing.Union[int, typing.Tuple[int, int]]
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

        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size[0:2]

        if isinstance(self.strides, int):
            strides = (self.strides, self.strides)
        else:
            strides = self.strides[0:2]

        conv = nn.Conv(
            features=self.features,
            kernel_size=kernel_size,
            strides=strides,
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
        branch1x1_conv = ConvBNReLU(
            features=64,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch1x1",
        )
        out_1x1 = branch1x1_conv(inputs, deterministic=m_deterministic)

        # branch 2: 1x1 convolution followed by 5x5 convolution
        branch5x5_1 = ConvBNReLU(
            features=48,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch5x5_1",
        )
        branch5x5_2 = ConvBNReLU(
            features=64,
            kernel_size=5,
            strides=1,
            padding=2,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch5x5_2",
        )
        out_5x5 = branch5x5_1(inputs, deterministic=m_deterministic)
        out_5x5 = branch5x5_2(out_5x5, deterministic=m_deterministic)

        # branch 3: 1x1 convolution followed by two 3x3 convolutions
        branch3x3_1 = ConvBNReLU(
            features=64,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_1",
        )
        branch3x3_2 = ConvBNReLU(
            features=96,
            kernel_size=3,
            strides=1,
            padding=1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_2",
        )
        branch3x3_3 = ConvBNReLU(
            features=96,
            kernel_size=3,
            strides=1,
            padding=1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_3",
        )
        out_3x3 = branch3x3_1(inputs, deterministic=m_deterministic)
        out_3x3 = branch3x3_2(out_3x3, deterministic=m_deterministic)
        out_3x3 = branch3x3_3(out_3x3, deterministic=m_deterministic)

        # branch 4: average pooling followed by 1x1 convolution
        branchpool = ConvBNReLU(
            features=self.pooled_features,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branchpool",
        )
        out_pool = nn.avg_pool(
            inputs,
            window_shape=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )
        out_pool = branchpool(out_pool, deterministic=m_deterministic)

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
        branch3x3 = ConvBNReLU(
            features=384,
            kernel_size=3,
            strides=2,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3",
        )
        out_3x3 = branch3x3(inputs, deterministic=m_deterministic)

        # branch 2: a cascade of three convolutions
        branch3x3dbl_1 = ConvBNReLU(
            features=64,
            kernel_size=1,
            strides=1,
            padding=0,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_1",
        )
        branch3x3dbl_2 = ConvBNReLU(
            features=96,
            kernel_size=3,
            strides=1,
            padding=1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_2",
        )
        branch3x3dbl_3 = ConvBNReLU(
            features=96,
            kernel_size=3,
            strides=2,
            padding=0,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_3",
        )
        out_3x3dbl = branch3x3dbl_1(inputs, deterministic=m_deterministic)
        out_3x3dbl = branch3x3dbl_2(out_3x3dbl, deterministic=m_deterministic)
        out_3x3dbl = branch3x3dbl_3(out_3x3dbl, deterministic=m_deterministic)

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


class InceptionCBlock(nn.Module):
    r"""An Inception block comprises factorized 7x7 convolution and avg pooling.

    Args:
        features (int): Dimentionality of 7x7 convolution output.
        deterministic (bool, optional): Whether to apply running averages
            in batch normalization.
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
    """

    features: int
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass of the Inception-C block.

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
        branch1x1_conv = ConvBNReLU(
            features=192,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch1x1",
        )
        out_1x1 = branch1x1_conv(inputs, deterministic=m_deterministic)

        # branch 2: factorized 7x7 convolution
        branch7x7_1 = ConvBNReLU(
            features=self.features,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch7x7_1",
        )
        branch7x7_2 = ConvBNReLU(
            features=self.features,
            kernel_size=(1, 7),
            strides=1,
            padding=(0, 3),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch7x7_2",
        )
        branch7x7_3 = ConvBNReLU(
            features=192,
            kernel_size=(7, 1),
            strides=1,
            padding=(3, 0),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch7x7_3",
        )
        out_7x7 = branch7x7_1(inputs, deterministic=m_deterministic)
        out_7x7 = branch7x7_2(out_7x7, deterministic=m_deterministic)
        out_7x7 = branch7x7_3(out_7x7, deterministic=m_deterministic)

        # branch 3: factorized 7x7 convolution (double)
        branch7x7dbl_1 = ConvBNReLU(
            features=self.features,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch7x7dbl_1",
        )
        branch7x7dbl_2 = ConvBNReLU(
            features=self.features,
            kernel_size=(7, 1),
            strides=1,
            padding=(3, 0),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch7x7dbl_2",
        )
        branch7x7dbl_3 = ConvBNReLU(
            features=self.features,
            kernel_size=(1, 7),
            strides=1,
            padding=(0, 3),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch7x7dbl_3",
        )
        branch7x7dbl_4 = ConvBNReLU(
            features=self.features,
            kernel_size=(7, 1),
            strides=1,
            padding=(3, 0),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch7x7dbl_4",
        )
        branch7x7dbl_5 = ConvBNReLU(
            features=192,
            kernel_size=(1, 7),
            strides=1,
            padding=(0, 3),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch7x7dbl_5",
        )
        out_7x7dbl = branch7x7dbl_1(inputs, deterministic=m_deterministic)
        out_7x7dbl = branch7x7dbl_2(out_7x7dbl, deterministic=m_deterministic)
        out_7x7dbl = branch7x7dbl_3(out_7x7dbl, deterministic=m_deterministic)
        out_7x7dbl = branch7x7dbl_4(out_7x7dbl, deterministic=m_deterministic)
        out_7x7dbl = branch7x7dbl_5(out_7x7dbl, deterministic=m_deterministic)

        # branch 4: average pooling followed by 1x1 convolution
        branchpool = ConvBNReLU(
            features=192,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branchpool",
        )
        out_pool = nn.avg_pool(
            inputs,
            window_shape=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )
        out_pool = branchpool(out_pool, deterministic=m_deterministic)

        return jnp.concatenate(
            [out_1x1, out_7x7, out_7x7dbl, out_pool],
            axis=-1,
        )


class InceptionDBlock(nn.Module):
    r"""An Inception block comprises 7x7, 3x3 convolutions and max pooling.

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
        self,
        inputs: jax.Array,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass of the Inception-D block.

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

        # branch 1: 1x1 convolution followed by 3x3 convolution with stride 2
        branch3x3_1 = ConvBNReLU(
            features=192,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3_1",
        )
        branch3x3_2 = ConvBNReLU(
            features=320,
            kernel_size=3,
            strides=2,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3_2",
        )
        out_3x3 = branch3x3_1(inputs, deterministic=m_deterministic)
        out_3x3 = branch3x3_2(out_3x3, deterministic=m_deterministic)

        # branch 2: factorized 7x7 convolutions
        branch7x7x3_1 = ConvBNReLU(
            features=192,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch7x7x3_1",
        )
        branch7x7x3_2 = ConvBNReLU(
            features=192,
            kernel_size=(1, 7),
            strides=1,
            padding=(0, 3),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch7x7x3_2",
        )
        branch7x7x3_3 = ConvBNReLU(
            features=192,
            kernel_size=(7, 1),
            strides=1,
            padding=(3, 0),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch7x7x3_3",
        )
        branch7x7x3_4 = ConvBNReLU(
            features=192,
            kernel_size=3,
            strides=2,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch7x7x3_4",
        )
        out_7x7x3 = branch7x7x3_1(inputs, deterministic=m_deterministic)
        out_7x7x3 = branch7x7x3_2(out_7x7x3, deterministic=m_deterministic)
        out_7x7x3 = branch7x7x3_3(out_7x7x3, deterministic=m_deterministic)
        out_7x7x3 = branch7x7x3_4(out_7x7x3, deterministic=m_deterministic)

        # branch 3: max pooling
        out_pool = nn.max_pool(
            inputs,
            window_shape=(3, 3),
            strides=(2, 2),
            padding="VALID",
        )

        return jnp.concatenate(
            [out_3x3, out_7x7x3, out_pool],
            axis=-1,
        )
