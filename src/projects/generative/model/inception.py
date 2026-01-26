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
        epsilon (float, optional): A small float added to avoid division by zero
            in batch normalization. Default is `0.001`.
        use_bias (bool): Whether to use bias in the convolution.
            Default is `False`.
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
    """

    features: int
    kernel_size: typing.Union[int, typing.Tuple[int, int]]
    strides: typing.Union[int, typing.Tuple[int, int]]
    padding: flax_typing.PaddingLike = "VALID"
    bn_use_bias: bool = True
    bn_use_scale: bool = False  # NOTE: aligns with TensorFlow implementation
    epsilon: float = 0.001  # NOTE: aligns with TensorFlow implementation
    use_bias: bool = False
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    def setup(self) -> None:
        r"""Instantiate the `Conv-BatchNorm-ReLU` block."""
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size[0:2]

        if isinstance(self.strides, int):
            strides = (self.strides, self.strides)
        else:
            strides = self.strides[0:2]

        self.conv = nn.Conv(
            features=self.features,
            kernel_size=kernel_size,
            strides=strides,
            padding=self.padding,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv",
        )

        self.batchnorm = nn.BatchNorm(
            epsilon=self.epsilon,
            use_bias=self.bn_use_bias,
            use_scale=self.bn_use_scale,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

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

        out = self.conv(inputs.astype(self.dtype))
        bn = nn.BatchNorm(
            use_running_average=m_deterministic,
            epsilon=0.001,
            momentum=0.1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="batchnorm",
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
            count_include_pad=False,
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
        features (int): Dimensionality of 7x7 convolution output.
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
        branch_pool = ConvBNReLU(
            features=192,
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
            count_include_pad=False,
        )
        out_pool = branch_pool(out_pool, deterministic=m_deterministic)

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


class InceptionEBlock(nn.Module):
    r"""An Inception block comprises factorized 3x3 convolutions and avg pool.

    Args:
        apply_max_pool (bool): Whether to apply max pooling.
        deterministic (bool, optional): Whether to apply running averages
            in batch normalization.
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
    """

    apply_max_pool: bool
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass of the Inception-E block.

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
            features=320,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch1x1",
        )
        out_1x1 = branch1x1_conv(inputs, deterministic=m_deterministic)

        # branch 2: divided 3x3 convolutions
        branch3x3_1 = ConvBNReLU(
            features=384,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3_1",
        )
        branch3x3_2a = ConvBNReLU(
            features=384,
            kernel_size=(1, 3),
            strides=1,
            padding=(0, 1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3_2a",
        )
        branch3x3_2b = ConvBNReLU(
            features=384,
            kernel_size=(3, 1),
            strides=1,
            padding=(1, 0),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3_2b",
        )
        out_3x3 = branch3x3_1(inputs, deterministic=m_deterministic)
        out_3x3_a = branch3x3_2a(out_3x3, deterministic=m_deterministic)
        out_3x3_b = branch3x3_2b(out_3x3, deterministic=m_deterministic)
        out_3x3 = jnp.concatenate([out_3x3_a, out_3x3_b], axis=-1)

        # branch 3: divided 3x3 convolutions (double)
        branch3x3dbl_1 = ConvBNReLU(
            features=448,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_1",
        )
        branch3x3dbl_2 = ConvBNReLU(
            features=384,
            kernel_size=3,
            strides=1,
            padding=1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_2",
        )
        branch3x3dbl_3a = ConvBNReLU(
            features=384,
            kernel_size=(1, 3),
            strides=1,
            padding=(0, 1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_3a",
        )
        branch3x3dbl_3b = ConvBNReLU(
            features=384,
            kernel_size=(3, 1),
            strides=1,
            padding=(1, 0),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch3x3dbl_3b",
        )
        out_3x3dbl = branch3x3dbl_1(inputs, deterministic=m_deterministic)
        out_3x3dbl = branch3x3dbl_2(out_3x3dbl, deterministic=m_deterministic)
        out_3x3dbl_a = branch3x3dbl_3a(
            out_3x3dbl,
            deterministic=m_deterministic,
        )
        out_3x3dbl_b = branch3x3dbl_3b(
            out_3x3dbl,
            deterministic=m_deterministic,
        )
        out_3x3dbl = jnp.concatenate([out_3x3dbl_a, out_3x3dbl_b], axis=-1)

        # branch 4: average pooling followed by 1x1 convolution
        branch_pool = ConvBNReLU(
            features=192,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="branch_pool",
        )
        if self.apply_max_pool:
            out_pool = nn.max_pool(
                inputs,
                window_shape=(3, 3),
                strides=(1, 1),
                padding="SAME",
            )
        else:
            out_pool = nn.avg_pool(
                inputs,
                window_shape=(3, 3),
                strides=(1, 1),
                padding="SAME",
                count_include_pad=False,
            )
        out_pool = branch_pool(out_pool, deterministic=m_deterministic)

        return jnp.concatenate(
            [out_1x1, out_3x3, out_3x3dbl, out_pool],
            axis=-1,
        )


class InceptionAuxiliaryHead(nn.Module):
    r"""Auxiliary classifier head for Inception architecture.

    Args:
        num_classes (int): Number of output classes.
        deterministic (bool, optional): Whether to apply running averages
            in batch normalization.
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
    """

    num_classes: int
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass of the Inception auxiliary block.

        Args:
            inputs (jax.Array): Input array of shape `(*, height, width, C)`.
            deterministic (bool, optional): Whether to apply running averages
                in batch normalization.

        Returns:
            Output array of shape `(*, num_classes)`.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )

        # average pooling
        out = nn.avg_pool(
            inputs,
            window_shape=(5, 5),
            strides=(3, 3),
            padding="VALID",
        )

        # 1x1 convolution
        conv_0 = ConvBNReLU(
            features=128,
            kernel_size=1,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv0",
        )
        out = conv_0(out, deterministic=m_deterministic)

        # 5x5 convolution
        conv_1 = ConvBNReLU(
            features=768,
            kernel_size=5,
            strides=1,
            padding=0,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv1",
        )
        out = conv_1(out, deterministic=m_deterministic)

        # average pooling over spatial dimensions
        out = jnp.mean(out, axis=(-3, -2), keepdims=True)
        out = jnp.reshape(out, (*out.shape[:-3], -1))

        # linear layer
        fc = nn.Dense(
            features=self.num_classes,
            use_bias=True,
            kernel_init=nn.initializers.variance_scaling(
                1.0,
                "fan_avg",
                "uniform",
            ),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="fc",
        )
        out = fc(out)

        return out


# ==============================================================================
# Inception Architecture
class InceptionV3(nn.Module):
    r"""Inception-v3 architecture.

    .. note::
        This module implements the Inception architecture from the original
        paper "Rethinking the Inception Architecture for Computer Vision"
        by Szegedy et al. (2015) at `https://arxiv.org/abs/1512.00567`.

    Args:
        num_classes (int): Number of output classes.
        last_block_max_pool (bool): Whether to apply max pooling in the last
            Inception-E block.
        with_head (Optional[bool], optional): Whether to include the final
            classification head. Default is `None`.
        with_aux_logits (Optional[bool], optional): Whether to include the
            auxiliary logits head. Default is `None`.
        deterministic (bool, optional): Whether to apply running averages
            in batch normalization.
        dtype (Any): The dtype of the computation.
        param_dtype (Any): The dtype of the parameters.
    """

    num_classes: int
    last_block_max_pool: bool
    with_head: typing.Optional[bool] = None
    with_aux_logits: typing.Optional[bool] = None
    deterministic: typing.Optional[bool] = None
    dtype: typing.Any = jnp.float32
    param_dtype: typing.Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        deterministic: typing.Optional[bool] = None,
        with_head: typing.Optional[bool] = None,
        with_aux_logits: typing.Optional[bool] = None,
    ) -> typing.Tuple[jax.Array, typing.Optional[jax.Array]]:
        r"""Forward pass of the Inception-v3 architecture.

        Args:
            inputs (jax.Array): Input array of shape `(*, height, width, 3)`.
            deterministic (bool, optional): Whether to apply running averages
                in batch normalization.
            with_head (bool, optional): Whether to include the final
                classification head. If `None`, uses the module's attribute.
            with_aux_logits (bool, optional): Whether to include the auxiliary
                logits head. If `None`, uses the module's attribute.

        Returns:
            If `with_head` is `False`, returns the feature map before the head;
            otherwise, returns a tuple of the final output array of shape
            `(*, num_classes)` and optionally the auxiliary logits of shape
            `(*, num_classes)` if `with_aux_logits` is `True`.
        """
        # sanity check
        assert (
            inputs.shape[-1] == 3
        ), f"Expected input with 3 channels (RGB), but got {inputs.shape[-1]}."

        # merge parameters
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )
        m_with_head = nn.merge_param(
            "with_head",
            self.with_head,
            with_head,
        )
        m_with_aux_logits = nn.merge_param(
            "with_aux_logits",
            self.with_aux_logits,
            with_aux_logits,
        )

        # stem blocks
        conv = ConvBNReLU(
            features=32,
            kernel_size=3,
            strides=2,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv",
        )
        out = conv(inputs, deterministic=m_deterministic)

        conv_1 = ConvBNReLU(
            features=32,
            kernel_size=3,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_1",
        )
        out = conv_1(out, deterministic=m_deterministic)

        conv_2 = ConvBNReLU(
            features=64,
            kernel_size=3,
            strides=1,
            padding=1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_2",
        )
        out = conv_2(out, deterministic=m_deterministic)

        out = nn.max_pool(
            out,
            window_shape=(3, 3),  # ksize = (1, 3, 3, 1)
            strides=(2, 2),  # strides = (1, 2, 2, 1)
            padding="VALID",
        )

        conv_3 = ConvBNReLU(
            features=80,
            kernel_size=1,
            strides=1,  # strides = (1, 1, 1, 1)
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_3",
        )
        out = conv_3(out, deterministic=m_deterministic)

        conv_4 = ConvBNReLU(
            features=192,
            kernel_size=3,
            strides=1,
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_4",
        )
        out = conv_4(out, deterministic=m_deterministic)

        out = nn.max_pool(
            out,
            window_shape=(3, 3),  # ksize = (1, 3, 3, 1)
            strides=(2, 2),  # strides = (1, 2, 2, 1)
            padding="VALID",
        )

        # inception blocks
        mixed = InceptionABlock(
            pooled_features=32,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mixed",
        )
        out = mixed(out, deterministic=m_deterministic)

        mixed_1 = InceptionABlock(
            pooled_features=64,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mixed_1",
        )
        out = mixed_1(out, deterministic=m_deterministic)

        mixed_2 = InceptionABlock(
            pooled_features=64,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mixed_2",
        )
        out = mixed_2(out, deterministic=m_deterministic)

        mixed_6a = InceptionBBlock(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mixed_6a",
        )
        mixed_6b = InceptionCBlock(
            features=128,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mixed_6b",
        )
        mixed_6c = InceptionCBlock(
            features=160,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mixed_6c",
        )
        mixed_6d = InceptionCBlock(
            features=160,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mixed_6d",
        )
        mixed_6e = InceptionCBlock(
            features=192,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mixed_6e",
        )
        out = mixed_6a(out, deterministic=m_deterministic)
        out = mixed_6b(out, deterministic=m_deterministic)
        out = mixed_6c(out, deterministic=m_deterministic)
        out = mixed_6d(out, deterministic=m_deterministic)
        out = mixed_6e(out, deterministic=m_deterministic)

        if m_with_aux_logits:
            aux_head = InceptionAuxiliaryHead(
                num_classes=self.num_classes,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="auxlogits",
            )
            aux_logits = aux_head(out, deterministic=m_deterministic)
        else:
            aux_logits = None

        mixed_7a = InceptionDBlock(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mixed_7a",
        )
        mixed_7b = InceptionEBlock(
            apply_max_pool=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mixed_7b",
        )
        mixed_7c = InceptionEBlock(
            apply_max_pool=self.last_block_max_pool,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mixed_7c",
        )
        out = mixed_7a(out, deterministic=m_deterministic)
        out = mixed_7b(out, deterministic=m_deterministic)
        out = mixed_7c(out, deterministic=m_deterministic)
        out = nn.avg_pool(
            out,
            window_shape=(8, 8),
            strides=(1, 1),
            padding="VALID",
            count_include_pad=False,
        )
        out = jnp.reshape(out, (*out.shape[:-3], 2048))
        if not m_with_head:
            return out, aux_logits

        # classification head
        dropout = nn.Dropout(rate=0.5)
        out = dropout(out, deterministic=m_deterministic)
        fc = nn.Dense(
            features=self.num_classes,
            use_bias=True,
            kernel_init=nn.initializers.variance_scaling(
                scale=1e-10,
                mode="fan_avg",
                distribution="uniform",
            ),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="output",
        )
        out = fc(out)

        return out, aux_logits
