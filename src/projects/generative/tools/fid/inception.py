import typing

from flax import linen as nn
from flax import typing as flax_typing
import jax


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
