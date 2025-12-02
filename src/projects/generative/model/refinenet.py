import typing

import chex
from flax import linen as nn
import jax
from jax._src import core as jax_core
from jax._src import dtypes as jax_dtypes
from jax._src import typing as jax_typing
import jax.numpy as jnp


# ==============================================================================
# Builder functions
# ==============================================================================
def _uniform_init() -> jax.nn.initializers.Initializer:
    r"""Uniform initializer for convolutional layers."""

    def init(
        key: jax.Array,
        shape: jax_typing.Shape,
        dtype: typing.Any = jnp.float_,
        out_sharding: typing.Any = None,
    ) -> jax.Array:
        """Uniform initializer for one-dimensional parameters."""
        dim = shape[-1]
        dtype = jax_dtypes.canonicalize_dtype(dtype)
        named_shape = jax_core.canonicalize_shape(shape)
        return jax.random.uniform(
            key=key,
            shape=named_shape,
            dtype=dtype,
            minval=-jnp.sqrt(1.0 / dim),
            maxval=jnp.sqrt(1.0 / dim),
            out_sharding=out_sharding,
        )

    return init


def _conv_1x1(
    out_channels: int,
    stride: int = 1,
    use_bias: bool = True,
    name: str = "conv1x1",
    dtype: typing.Any = jnp.float32,
    param_dtype: typing.Any = jnp.float32,
) -> nn.Conv:
    r"""1x1 convolution with stride and padding."""
    return nn.Conv(
        features=out_channels,
        kernel_size=(1, 1),
        strides=(stride, stride),
        padding=(0, 0),
        use_bias=use_bias,
        kernel_init=jax.nn.initializers.variance_scaling(
            scale=1.0,
            mode="fan_in",
            distribution="uniform",
        ),
        bias_init=_uniform_init(),
        name=name,
        dtype=dtype,
        param_dtype=param_dtype,
    )


def _conv_3x3(
    out_channels: int,
    stride: int = 1,
    use_bias: bool = True,
    name: str = "conv3x3",
    dtype: typing.Any = jnp.float32,
    param_dtype: typing.Any = jnp.float32,
) -> nn.Conv:
    r"""3x3 convolution with stride and padding."""
    return nn.Conv(
        features=out_channels,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding=(1, 1),
        use_bias=use_bias,
        kernel_init=jax.nn.initializers.variance_scaling(
            scale=1.0,
            mode="fan_in",
            distribution="uniform",
        ),
        bias_init=_uniform_init(),
        name=name,
        dtype=dtype,
        param_dtype=param_dtype,
    )


def _dilated_conv_3x3(
    out_channels: int,
    dilation: int,
    use_bias: bool = True,
    name: str = "dilated_conv3x3",
    dtype: typing.Any = jnp.float32,
    param_dtype: typing.Any = jnp.float32,
) -> nn.Conv:
    r"""3x3 dilated convolution with dilation and padding."""
    return nn.Conv(
        features=out_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=(dilation, dilation),
        kernel_dilation=(dilation, dilation),
        use_bias=use_bias,
        kernel_init=jax.nn.initializers.variance_scaling(
            scale=1.0,
            mode="fan_in",
            distribution="uniform",
        ),
        bias_init=_uniform_init(),
        name=name,
        dtype=dtype,
        param_dtype=param_dtype,
    )


# ==============================================================================
# Layers
# ==============================================================================
class ConditionalInstanceNorm2dPlus(nn.Module):
    r"""Conditional Instance Normalization with extra affine transformation."""

    features: int
    """int: Dimensionality of the feature map."""
    num_classes: int
    """int: Number of conditioning classes."""
    use_bias: bool = True
    """bool: If True, add bias after the normalization."""
    dtype: typing.Any = jnp.float32
    """dtype: The data type of the computation (default: float32)."""
    param_dtype: typing.Any = jnp.float32
    """param_dtype: The data type of the parameters (default: float32)."""

    def setup(self) -> None:
        """Instantiate a `ConditionalInstanceNorm2dPlus` module."""

        self.instance_norm = nn.LayerNorm(
            reduction_axes=(-3, -2),
            feature_axes=-1,
            use_bias=False,
            use_scale=False,
            epsilon=1e-5,
            name="instance_norm",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        def _kernel_init(
            key: typing.Any,
            shape: jax_typing.Shape,
            dtype: typing.Any,
            out_sharding: typing.Any = None,
        ) -> jax.Array:
            dtype = jax_dtypes.canonicalize_dtype(dtype)
            named_shape = jax_core.canonicalize_shape(shape)
            return (
                1.0
                + jax.random.normal(
                    key=key,
                    shape=named_shape,
                    dtype=dtype,
                    out_sharding=out_sharding,
                )
                * 0.02
            )

        if self.use_bias:
            self.embed = nn.Embed(
                self.num_classes,
                self.features * 3,
                embedding_init=_kernel_init,
                name="embed",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        else:
            self.embed = nn.Embed(
                self.num_classes,
                self.features * 2,
                embedding_init=_kernel_init,
                name="embed",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

    def __call__(self, inputs: jax.Array, cond: jax.Array) -> jax.Array:
        """Forward pass of the `ConditionalInstanceNorm2dPlus` module.

        Args:
            inputs (jax.Array): Input feature map of shape `(*, H, W, C)`.
            cond (jax.Array): Condition feature map of shape `(*, )`.

        Returns:
            Output feature map of shape `(*, H, W, C)`.
        """
        batch_dims = inputs.shape[:-3]
        chex.assert_shape(cond, (*batch_dims,))

        means = jnp.mean(inputs, axis=(-3, -2), keepdims=False)
        m = jnp.mean(means, axis=-1, keepdims=True)
        v = jnp.var(means, axis=-1, keepdims=True)
        means = jnp.true_divide((means - m), jnp.sqrt(v + 1e-5))
        means = means[..., None, None, :]
        h = self.instance_norm(inputs)

        if self.use_bias:
            gamma, alpha, beta = jnp.split(self.embed(cond), 3, axis=-1)
            gamma = jnp.expand_dims(gamma, axis=(-3, -2))
            alpha = jnp.expand_dims(alpha, axis=(-3, -2))
            beta = jnp.expand_dims(beta, axis=(-3, -2))
            chex.assert_equal_rank((h, gamma, alpha, beta))
            h = h + means * jnp.broadcast_to(alpha, h.shape)
            output = jnp.add(
                jnp.broadcast_to(gamma, h.shape) * h,
                jnp.broadcast_to(beta, h.shape),
            )
        else:
            gamma, alpha = jnp.split(self.embed(cond), 2, axis=-1)
            gamma = jnp.expand_dims(gamma, axis=(-3, -2))
            alpha = jnp.expand_dims(alpha, axis=(-3, -2))
            chex.assert_equal_rank((h, gamma, alpha))
            h = h + means * jnp.broadcast_to(alpha, h.shape)
            output = jnp.broadcast_to(gamma, h.shape) * h

        return output


class ConvMeanPool(nn.Module):
    r"""Convolution followed by average pooling."""

    features: int
    """int: Number of output channels."""
    kernel_size: int = 3
    """int: Size of the convolutional kernel (default: 3)."""
    adjust_padding: bool = False
    """bool: If True, adjust padding for even-sized kernels (default: False)."""
    dtype: typing.Any = jnp.float32
    """dtype: The data type of the computation (default: float32)."""
    param_dtype: typing.Any = jnp.float32
    """param_dtype: The data type of the parameters (default: float32)."""

    def setup(self) -> None:
        """Instantiate a ConvMeanPool module."""
        self.conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(1, 1),
            padding=(self.kernel_size // 2, self.kernel_size // 2),
            name="conv",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        r"""Forward pass of the `ConvMeanPool` module.

        Args:
            inputs (jax.Array): Input feature map of shape `(*, H, W, C)`.

        Returns:
            Output feature map of shape `(*, H/2, W/2, C_out)`.
        """
        batch_dims = inputs.shape[:-3]
        if self.adjust_padding:
            inputs = jnp.pad(
                inputs,
                [(0, 0)] * len(batch_dims) + [(1, 0), (1, 0), (0, 0)],
                mode="constant",
                constant_values=0,
            )
        output = self.conv(inputs)
        output = nn.avg_pool(
            inputs=output,
            window_shape=(2, 2),
            strides=(2, 2),
            padding="VALID",
        )
        return output


# ==============================================================================
# Modules
# ==============================================================================
class ConditionalResidualBlock(nn.Module):
    r"""Residual block with conditioning feature map."""

    in_channels: int
    """int: Number of channels of the input feature map."""
    out_channels: int
    """int: Number of channels of the output feature map."""
    norm_module: typing.Callable[..., nn.Module]
    """Callable[..., nn.Module]: Normalization module to use."""
    dilation: typing.Optional[int] = None
    """Optional[int]: Optional dilations in the convolutional layers."""
    resample: typing.Optional[str] = None
    """Optional[str]: Resampling method, either `down`, or None."""
    adjusting_padding: bool = False
    """bool: If True, adjust padding for even-sized kernels (default: False)."""
    dtype: typing.Any = jnp.float32
    """dtype: The data type of the computation (default: float32)."""
    param_dtype: typing.Any = jnp.float32
    """param_dtype: The data type of the parameters (default: float32)."""

    def setup(self) -> None:
        r"""Instantiate a conditional residual block."""
        self.norm_1 = self.norm_module(
            features=self.in_channels,
            name="normalize1",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        if self.resample == "down":
            if self.dilation is not None:
                self.conv_1 = _dilated_conv_3x3(
                    self.in_channels,
                    dilation=self.dilation,
                    name="conv1",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                self.norm_2 = self.norm_module(
                    features=self.in_channels,
                    name="normalize2",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                self.conv_2 = _dilated_conv_3x3(
                    self.out_channels,
                    dilation=self.dilation,
                    name="conv2",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                self.shortcut_conv = _dilated_conv_3x3(
                    self.out_channels,
                    dilation=self.dilation,
                    name="shortcut",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            else:
                self.conv_1 = _conv_3x3(
                    self.in_channels,
                    stride=1,
                    name="conv1",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                self.norm_2 = self.norm_module(
                    features=self.in_channels,
                    name="normalize2",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                self.conv_2 = ConvMeanPool(
                    features=self.out_channels,
                    kernel_size=3,
                    adjust_padding=self.adjusting_padding,
                    name="conv2",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                self.shortcut_conv = ConvMeanPool(
                    features=self.out_channels,
                    kernel_size=1,
                    adjust_padding=self.adjusting_padding,
                    name="shortcut",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
        elif self.resample is None:
            if self.dilation is not None:
                self.conv_1 = _dilated_conv_3x3(
                    self.out_channels,
                    dilation=self.dilation,
                    name="conv1",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                self.norm_2 = self.norm_module(
                    features=self.out_channels,
                    name="normalize2",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                self.conv_2 = _dilated_conv_3x3(
                    self.out_channels,
                    dilation=self.dilation,
                    name="conv2",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                self.shortcut_conv = _dilated_conv_3x3(
                    self.out_channels,
                    dilation=self.dilation,
                    name="shortcut",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            else:
                self.conv_1 = _conv_3x3(
                    self.out_channels,
                    name="conv1",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                self.norm_2 = self.norm_module(
                    features=self.out_channels,
                    name="normalize2",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                self.conv_2 = _conv_3x3(
                    self.out_channels,
                    name="conv2",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                self.shortcut_conv = _conv_1x1(
                    self.out_channels,
                    name="shortcut",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
        else:
            raise ValueError(f"Invalid resample: {self.resample}")

        self.use_shortcut = (self.in_channels != self.out_channels) or (
            self.resample is not None
        )

    def __call__(self, inputs: jax.Array, cond: jax.Array) -> jax.Array:
        r"""Forward pass of the conditional residual block.

        Args:
            inputs (jax.Array): Input feature map of shape `(*, H, W, D)`.
            cond (jax.Array): Condition feature map of shape `(*,)`.

        Returns:
            Output feature map of shape `(*, H, W, D)`.
        """
        output = self.norm_1(inputs, cond)
        output = jax.nn.elu(output)
        output = self.conv_1(output)
        output = self.norm_2(output, cond)
        output = jax.nn.elu(output)
        output = self.conv_2(output)

        if self.use_shortcut:
            shortcut = self.shortcut_conv(inputs)
        else:
            shortcut = inputs
        chex.assert_equal_shape((output, shortcut))

        return output + shortcut


class ConditionalRCUBlock(nn.Module):
    r"""Refinement Convolution Unit (RCU) block with conditioning features."""

    features: int
    """int: Dimensionality of the feature map."""
    norm_module: typing.Callable[..., nn.Module]
    """Callable[..., nn.Module]: Normalization module to use."""
    num_blocks: int
    """int: Number of repeated blocks in the cascade."""
    num_stages: int
    """int: Number of stages in each block."""
    dtype: typing.Any = jnp.float32
    """dtype: The data type of the computation (default: float32)."""
    param_dtype: typing.Any = jnp.float32
    """param_dtype: The data type of the parameters (default: float32)."""

    def setup(self) -> None:
        r"""Instantiate a `ConditionalRCUBlock` module."""
        convs, norms = [], []
        for i in range(self.num_blocks):
            for j in range(self.num_stages):
                convs.append(
                    _conv_3x3(
                        out_channels=self.features,
                        stride=1,
                        use_bias=False,
                        name=f"{i+1}_{j+1}_conv",
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    )
                )
                norms.append(
                    self.norm_module(
                        features=self.features,
                        use_bias=True,
                        name=f"{i+1}_{j+1}_norm",
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                    )
                )
        self.convs = convs
        self.norms = norms

    def __call__(self, inputs: jax.Array, cond: jax.Array) -> jax.Array:
        r"""Forward pass of the `ConditionalRCUBlock` module.

        Args:
            inputs (jax.Array): Input feature map of shape `(*, H, W, C)`.
            cond (jax.Array): Condition feature map of shape `(*, H, W, d)`.

        Returns:
            Output feature map of shape `(*, H, W, C)`.
        """
        _idx: int = 0
        output = inputs
        for _ in range(self.num_blocks):
            residual = output
            for _ in range(self.num_stages):
                output = self.norms[_idx](inputs=output, cond=cond)
                output = jax.nn.elu(output)
                output = self.convs[_idx](inputs=output)
                _idx += 1
            output = output + residual

        return output


class ConditionalMSFBlock(nn.Module):
    r"""Conditional Multi-Scale Feature block."""

    in_features: typing.Sequence[int]
    """Sequence[int]: List of input feature map dimensionalities."""
    features: int
    """int: Dimensionality of the output feature map."""
    norm_module: typing.Callable[..., nn.Module]
    """Callable[..., nn.Module]: Normalization module to use."""
    dtype: typing.Any = jnp.float32
    """dtype: The data type of the computation (default: float32)."""
    param_dtype: typing.Any = jnp.float32
    """param_dtype: The data type of the parameters (default: float32)."""

    def setup(self) -> None:
        r"""Instantiate a `ConditionalMSFBlock` module."""
        convs, norms = [], []
        for i, in_feature in enumerate(self.in_features):
            convs.append(
                _conv_3x3(
                    out_channels=self.features,
                    stride=1,
                    use_bias=True,
                    name=f"convs.{i:d}",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            )
            norms.append(
                self.norm_module(
                    features=in_feature,
                    use_bias=True,
                    name=f"norms.{i:d}",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            )
        self.convs = convs
        self.norms = norms

    def __call__(
        self,
        inputs: typing.Sequence[jax.Array],
        cond: jax.Array,
        shape: jax_typing.Shape,
    ) -> jax.Array:
        r"""Forward pass of the `ConditionalMSFBlock` module.

        Args:
            inputs (Sequence[jax.Array]): Sequence of input feature maps to be
                merged. Each feature map has shape `(*, H_i, W_i, C)
            cond (jax.Array): Condition feature map of shape `(*, H, W, d)`.
            shape (jax._src.typing.Shape): Shape of the output feature map.

        Returns:
            Output feature map of shape `(*, H, W, C)`.
        """
        assert isinstance(inputs, typing.Sequence) and len(inputs) == len(
            self.in_features
        ), (
            f"`inputs` must be a sequence of length {len(self.in_features)}, "
            f"but got {len(inputs)}.",
        )
        out_shape = (*inputs[0].shape[:-3], *shape, self.features)
        output = jnp.zeros(shape=out_shape, dtype=self.dtype)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = norm(inputs=inputs[i], cond=cond)
            h = conv(inputs=h)
            # NOTE: `jax.image.resize` aligns corners by default
            # see https://github.com/google/flax/discussions/2211
            h = jax.image.resize(h, shape=out_shape, method="bilinear")
            output = output + h
        return output


class ConditionalCRPBlock(nn.Module):
    r"""Conditional convolutional residual pooling (CRP) block."""

    features: int
    """int: Dimensionality of the output feature map."""
    norm_module: typing.Callable[..., nn.Module]
    """Callable[..., nn.Module]: Normalization module to use."""
    num_stages: int
    """int: Number of stages in the cascade."""
    dtype: typing.Any = jnp.float32
    """dtype: The data type of the computation (default: float32)."""
    param_dtype: typing.Any = jnp.float32
    """param_dtype: The data type of the parameters (default: float32)."""

    def setup(self) -> None:
        r"""Instantiate a `ConditionalCRPBlock` module."""
        convs, norms = [], []
        for i in range(self.num_stages):
            convs.append(
                _conv_3x3(
                    out_channels=self.features,
                    stride=1,
                    use_bias=False,
                    name=f"convs.{i:d}",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            )
            norms.append(
                self.norm_module(
                    features=self.features,
                    use_bias=True,
                    name=f"norms.{i:d}",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            )
        self.convs = convs
        self.norms = norms

    def __call__(self, inputs: jax.Array, cond: jax.Array) -> jax.Array:
        r"""Forward pass of the `ConditionalCRPBlock` module.

        Args:
            inputs (jax.Array): Input feature map of shape `(*, H, W, C)`.
            cond (jax.Array): Condition feature map of shape `(*, H, W, d)`.

        Returns:
            Output feature map of shape `(*, H, W, C)`.
        """
        output = jax.nn.elu(inputs)
        path = output
        for conv, norm in zip(self.convs, self.norms):
            path = norm(path, cond)
            path = nn.avg_pool(
                inputs=path,
                window_shape=(5, 5),
                strides=(1, 1),
                padding=((2, 2), (2, 2)),  # type: ignore
            )
            path = conv(path)
            output = output + path
        return output


class ConditionalRefineBlock(nn.Module):
    r"""Refinement block with skip connections and conditioning feature map."""

    in_features: typing.Sequence[int]
    """Sequence[int]: List of input feature map dimensionalities."""
    out_features: int
    """int: Number of output channels of each convolution."""
    norm_module: typing.Callable[[typing.Any], nn.Module]
    """Callable[Any, nn.Module]: Normalization module to use."""
    is_last_block: bool = False
    """bool: If True, this is the last refinement block."""
    dtype: typing.Any = jnp.float32
    """dtype: The data type of the computation (default: float32)."""
    param_dtype: typing.Any = jnp.float32
    """param_dtype: The data type of the parameters (default: float32)."""

    def setup(self) -> None:
        """Instantiate a refinement block."""
        adapt_convs = []
        for i, in_feature in enumerate(self.in_features):
            adapt_convs.append(
                ConditionalRCUBlock(
                    features=in_feature,
                    num_blocks=2,
                    num_stages=2,
                    norm_module=self.norm_module,
                    name=f"adapt_convs.{i:d}",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            )
        self.adapt_convs: typing.List[ConditionalRCUBlock] = adapt_convs
        self.output_convs = ConditionalRCUBlock(
            features=self.out_features,
            norm_module=self.norm_module,
            num_blocks=3 if self.is_last_block else 1,
            num_stages=2,
            name="output_convs",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        if len(self.in_features) > 1:
            self.msf = ConditionalMSFBlock(
                in_features=self.in_features,
                features=self.out_features,
                norm_module=self.norm_module,
                name="msf",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        self.crp = ConditionalCRPBlock(
            features=self.out_features,
            norm_module=self.norm_module,
            num_stages=2,
            name="crp",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        inputs: typing.List[jax.Array],
        cond: jax.Array,
        output_shape: jax_typing.Shape,
    ) -> jax.Array:
        r"""Forward pass of the refinement block.

        Args:
            inputs (List[jax.Array]): List of input feature maps to be merged.
                Each feature map has shape `(*, H_i, W_i, C)`.
            cond (jax.Array): Condition feature map of shape `(*, H, W, d)`.
            output_shape (jax._src.typing.Shape): Shape of the output feature.

        Returns:
            Output feature map of shape `(*, H, W, 128)`.
        """
        assert (
            isinstance(inputs, typing.Sequence)
            and len(inputs) == len(self.in_features)
            and all(
                inp.shape[-1] == self.in_features[i]
                for i, inp in enumerate(inputs)
            )
        ), (
            f"`inputs` must be a sequence of length {len(self.in_features)}, "
            f"with feature dimensions {self.in_features}, "
            f"but got {[inp.shape for inp in inputs]}.",
        )
        hs = []
        for i, x in enumerate(inputs):
            h = self.adapt_convs[i](inputs=x, cond=cond)
            hs.append(h)

        if len(self.in_features) > 1:
            h = self.msf(inputs=hs, cond=cond, shape=output_shape)
        else:
            h = hs[0]

        output = self.crp(inputs=h, cond=cond)
        output = self.output_convs(inputs=output, cond=cond)

        return output


# ==============================================================================
# Models
# ==============================================================================
class ConditionalRefineNet(nn.Module):
    r"""Multi-path Refinement Network with Conditional Instance Normlization.

    This module is adapted from the original implementation of
    `CondRefineNetDeeperDilated` in the NCSN official repository:
    `https://github.com/ermongroup/ncsn/blob/master/models/cond_refinenet_dilated.py`

    Attributes:
        in_channels (int): Number of channels of the input feature map.
        image_size (int): Size of the input (square) image.
            By default, it only supports `28` or `32`.
        latent_channels (int): Number of channels of the latent feature map.
        norm_module (Type[nn.Module]): Normalization module to use.
        dtype (dtype): The data type of the computation (default: float32).
        param_dtype (dtype): The data type of the parameters (default: float32).
    """

    in_channels: int
    """int: Number of channels of the input feature map."""
    image_size: int
    """int: Size of the input (square) image, either `28` or `32`."""
    latent_channels: int
    """int: Number of channels of the latent feature map."""
    norm_module: typing.Callable[..., nn.Module]
    """Callable[..., nn.Module]: Normalization module to use."""
    dtype: typing.Any = jnp.float32
    """dtype: The data type of the computation (default: float32)."""
    param_dtype: typing.Any = jnp.float32
    """param_dtype: The data type of the parameters (default: float32)."""

    def setup(self) -> None:
        """Instantiate a Refinement Network module."""
        if self.image_size not in [28, 32]:
            raise ValueError(
                "`image_size` must be either `28` or `32`, "
                f"but got {self.image_size}."
            )

        self.conv_in = _conv_3x3(
            out_channels=self.latent_channels,
            stride=1,
            use_bias=True,
            name="begin_conv",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        # build the residual blocks
        self.res_1 = [
            ConditionalResidualBlock(
                in_channels=self.latent_channels,
                out_channels=self.latent_channels,
                norm_module=self.norm_module,
                resample=None,
            ),
            ConditionalResidualBlock(
                in_channels=self.latent_channels,
                out_channels=self.latent_channels,
                norm_module=self.norm_module,
                resample=None,
            ),
        ]
        self.res_2 = [
            ConditionalResidualBlock(
                in_channels=self.latent_channels,
                out_channels=self.latent_channels * 2,
                norm_module=self.norm_module,
                resample="down",
            ),
            ConditionalResidualBlock(
                in_channels=self.latent_channels * 2,
                out_channels=self.latent_channels * 2,
                norm_module=self.norm_module,
                resample=None,
            ),
        ]
        self.res_3 = [
            ConditionalResidualBlock(
                in_channels=self.latent_channels * 2,
                out_channels=self.latent_channels * 2,
                norm_module=self.norm_module,
                resample="down",
            ),
            ConditionalResidualBlock(
                in_channels=self.latent_channels * 2,
                out_channels=self.latent_channels * 2,
                norm_module=self.norm_module,
                resample=None,
            ),
        ]
        if self.image_size == 28:
            self.res_4 = [
                ConditionalResidualBlock(
                    in_channels=self.latent_channels * 2,
                    out_channels=self.latent_channels * 2,
                    norm_module=self.norm_module,
                    resample="down",
                    adjusting_padding=True,
                    dilation=4,
                ),
                ConditionalResidualBlock(
                    in_channels=self.latent_channels * 2,
                    out_channels=self.latent_channels * 2,
                    norm_module=self.norm_module,
                    resample=None,
                    adjusting_padding=False,
                    dilation=4,
                ),
            ]
        elif self.image_size == 32:
            self.res_4 = [
                ConditionalResidualBlock(
                    in_channels=self.latent_channels * 2,
                    out_channels=self.latent_channels * 2,
                    norm_module=self.norm_module,
                    resample="down",
                    adjusting_padding=False,
                    dilation=4,
                ),
                ConditionalResidualBlock(
                    in_channels=self.latent_channels * 2,
                    out_channels=self.latent_channels * 2,
                    norm_module=self.norm_module,
                    resample=None,
                    adjusting_padding=False,
                    dilation=4,
                ),
            ]
        else:
            raise ValueError(
                "`image_size` must be either `28` or `32`, "
                f"but got {self.image_size}."
            )

        # build the refinement blocks
        self.refine_1 = ConditionalRefineBlock(
            in_features=[2 * self.latent_channels],
            out_features=2 * self.latent_channels,
            norm_module=self.norm_module,
            name="refine1",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.refine_2 = ConditionalRefineBlock(
            in_features=[2 * self.latent_channels, 2 * self.latent_channels],
            out_features=2 * self.latent_channels,
            norm_module=self.norm_module,
            name="refine2",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.refine_3 = ConditionalRefineBlock(
            in_features=[2 * self.latent_channels, 2 * self.latent_channels],
            out_features=self.latent_channels,
            norm_module=self.norm_module,
            name="refine3",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.refine_4 = ConditionalRefineBlock(
            in_features=[self.latent_channels, self.latent_channels],
            out_features=self.latent_channels,
            norm_module=self.norm_module,
            is_last_block=True,
            name="refine4",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        # final normalization and convolution
        self.norm = self.norm_module(
            features=self.latent_channels,
            name="normalizer",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.conv_out = _conv_3x3(
            out_channels=self.in_channels,
            stride=1,
            use_bias=True,
            name="end_conv",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        inputs: jax.Array,
        cond: jax.Array,
        **kwargs,  # type: ignore[unused-argument]
    ) -> jax.Array:
        r"""Forward pass of the conditional refinement network.

        Args:
            inputs (jax.Array): Input feature map of shape `(*, H, W, C)`.
            cond (jax.Array): Condition feature map of shape `(*,)`.

        Returns:
            Output feature map of shape `(*, H, W, C)`.
        """
        batch_dims = inputs.shape[:-3]
        dims = chex.Dimensions(
            H=self.image_size,
            W=self.image_size,
            D=self.in_channels,
        )
        chex.assert_shape(inputs, (*batch_dims, *dims["HWD"]))

        output = self.conv_in(inputs)

        # intermediate layers
        layer_1_out = self._forward_cond_res_block(
            module=self.res_1,
            inputs=output,
            cond=cond,
        )
        layer_2_out = self._forward_cond_res_block(
            module=self.res_2,
            inputs=layer_1_out,
            cond=cond,
        )
        layer_3_out = self._forward_cond_res_block(
            module=self.res_3,
            inputs=layer_2_out,
            cond=cond,
        )
        layer_4_out = self._forward_cond_res_block(
            module=self.res_4,
            inputs=layer_3_out,
            cond=cond,
        )

        # add conditional instance normalization
        refine_1_out = self.refine_1(
            inputs=[layer_4_out],
            cond=cond,
            output_shape=layer_4_out.shape[-3:-1],
        )
        refine_2_out = self.refine_2(
            inputs=[layer_3_out, refine_1_out],
            cond=cond,
            output_shape=layer_3_out.shape[-3:-1],
        )
        refine_3_out = self.refine_3(
            inputs=[layer_2_out, refine_2_out],
            cond=cond,
            output_shape=layer_2_out.shape[-3:-1],
        )
        refine_4_out = self.refine_4(
            inputs=[layer_1_out, refine_3_out],
            cond=cond,
            output_shape=layer_1_out.shape[-3:-1],
        )

        output = self.norm(refine_4_out, cond)
        output = jax.nn.elu(output)
        output = self.conv_out(output)
        chex.assert_shape(output, (*batch_dims, *dims["HWD"]))

        return output

    @staticmethod
    def _forward_cond_res_block(
        module: typing.Sequence[nn.Module],
        inputs: jax.Array,
        cond: jax.Array,
    ) -> jax.Array:
        r"""Forward pass through a residual block with conditional inputs."""
        for m in module:
            assert isinstance(m, ConditionalResidualBlock)
            inputs = m(inputs=inputs, cond=cond)
        return inputs
