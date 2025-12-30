import functools
import math
import typing

import chex
from flax import linen as nn
import jax
from jax import numpy as jnp


# ==============================================================================
# Initializers
@typing.runtime_checkable
class DefaultInitializer(typing.Protocol):
    def __call__(
        self,
        key: jax.Array,
        shape: typing.Sequence[typing.Union[int, typing.Any]],
        dtype: typing.Any,
        fan_in: int,
        fan_out: int,
    ) -> jax.Array:
        raise NotImplementedError


def default_init(
    scale: float = 1.0,
    mode: str = "xavier_uniform",
) -> DefaultInitializer:
    r"""Returns a parameter initializer with given scaling."""

    def xavier_uniform_init(
        key: jax.Array,
        shape: typing.Sequence[typing.Union[int, typing.Any]],
        dtype: typing.Any,
        fan_in: int,
        fan_out: int,
    ) -> jax.Array:
        limit = scale * jnp.sqrt(6.0 / (fan_in + fan_out))
        sample = jax.random.uniform(key=key, shape=shape, dtype=dtype)
        return limit * (sample * 2 - 1)

    def xavier_normal_init(
        key: jax.Array,
        shape: typing.Sequence[typing.Union[int, typing.Any]],
        dtype: typing.Any,
        fan_in: int,
        fan_out: int,
    ) -> jax.Array:
        std = scale * jnp.sqrt(2.0 / (fan_in + fan_out))
        return std * jax.random.normal(key=key, shape=shape, dtype=dtype)

    def kaiming_uniform_init(
        key: jax.Array,
        shape: typing.Sequence[typing.Union[int, typing.Any]],
        dtype: typing.Any,
        fan_in: int,
        fan_out: int,
    ) -> jax.Array:
        del fan_out
        limit = scale * jnp.sqrt(3.0 / fan_in)
        sample = jax.random.uniform(key=key, shape=shape, dtype=dtype)
        return limit * (sample * 2 - 1)

    def kaiming_normal_init(
        key: jax.Array,
        shape: typing.Sequence[typing.Union[int, typing.Any]],
        dtype: typing.Any,
        fan_in: int,
        fan_out: int,
    ) -> jax.Array:
        del fan_out
        std = scale * jnp.sqrt(1.0 / fan_in)
        return std * jax.random.normal(key=key, shape=shape, dtype=dtype)

    if mode == "xavier_uniform":
        return xavier_uniform_init
    elif mode == "xavier_normal":
        return xavier_normal_init
    elif mode == "kaiming_uniform":
        return kaiming_uniform_init
    elif mode == "kaiming_normal":
        return kaiming_normal_init
    else:
        raise ValueError(f"Unsupported initialization mode: {mode}.")


# ==============================================================================
# Modules
class Conv2D(nn.Module):
    r"""Two-dimensional convolutional layer with optional up/downsampling.

    Args:
        features (int): Number of output features.
        kernel_size (Optional[int], optional): Size of the convolutional kernel.
            If `None`, will not apply convolution. Default is `None`.
        use_bias (bool, optional): If `True`, uses bias in the convolution.
            Default is `True`.
        kernel_init (Callable, optional): Kernel initializer for the
            convolutional layer. Default is `_default_init(scale=1.0)`.
        bias_init (Callable, optional): Bias initializer for the convolutional
            layer. Default is `_default_init(scale=1e-10)`.
        resample_filter (typing.Sequence[typing.Union[float, int]], optional):
            One-dimensional FIR filter for resampling. Default is `[1, 1]`.
        downsampling (bool, optional): If `True`, applies a `scale=2`
            downsampling before convolution. Default is `False`.
        upsampling (bool, optional): If `True`, applies a `scale=2`
            upsampling before downsampling. Default is `False`.
        init_weight (float, optional): Element-wise scaling factor for the
            convolutional kernel initialization. Default is :math:`1.0`.
        init_bias (float, optional): Element-wise scaling factor for the
            convolutional bias initialization. Default is :math:`0.0`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    features: int
    kernel_size: typing.Optional[int] = None
    use_bias: bool = True
    kernel_init: DefaultInitializer = default_init(scale=1.0)
    bias_init: DefaultInitializer = default_init(scale=1e-10)
    resample_filter: typing.Sequence[typing.Union[float, int]] = (1, 1)
    downsampling: bool = False
    upsampling: bool = False
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        r"""Forward pass of the `Conv2D` layer.

        Args:
            inputs (jax.Array): Input array of shape `(*, H, W, C_in)`.

        Returns:
            Output array of shape `(*, H, W, C_out)` if not up/downsampling;
                otherwise shape is `(2 * H, 2 * W)` after upsampling or
                `(H / 2, W / 2)` after downsampling.
        """
        batch_dims = inputs.shape[:-3]
        height, width, channels = inputs.shape[-3:]
        out = inputs.reshape(-1, height, width, channels)

        # preprocess the resample filter
        f = jnp.array(self.resample_filter, dtype=out.dtype)
        chex.assert_rank(f, 1)
        f = jnp.outer(f, f) / jnp.sum(jnp.square(f))  # shape: [k, k]

        # applies upsampling if specified
        if self.upsampling:
            f_pad_left = f.shape[-1] // 2
            f_pad_right = f.shape[-1] - f_pad_left
            out = jax.lax.conv_general_dilated(
                lhs=out,
                lhs_dilation=(2, 2),
                # NOTE: for upsampling, multiply filter by 4 to preserve signal
                rhs=jnp.tile(f[:, :, None, None] * 4, (1, 1, 1, channels)),
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
                feature_group_count=channels,
                window_strides=(1, 1),
                padding=[(f_pad_left, f_pad_right), (f_pad_left, f_pad_right)],
                precision=self.precision,
            )

        # applies downsampling if specified
        if self.downsampling:
            f_pad = (f.shape[-1] - 1) // 2
            out = jax.lax.conv_general_dilated(
                lhs=out,
                rhs=jnp.tile(f[:, :, None, None], (1, 1, 1, channels)),
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
                feature_group_count=channels,
                window_strides=(2, 2),
                padding=[(f_pad, f_pad), (f_pad, f_pad)],
                precision=self.precision,
            )

        # applies convolution
        if isinstance(self.kernel_size, int):
            shp = [self.kernel_size, self.kernel_size, channels, self.features]
            kernel = self.param(
                "kernel",
                self.kernel_init,
                shp,
                self.param_dtype,
                channels * self.kernel_size * self.kernel_size,
                self.features * self.kernel_size * self.kernel_size,
            )
            padding = self.kernel_size // 2
            out = jax.lax.conv_general_dilated(
                lhs=out,
                rhs=kernel,
                window_strides=(1, 1),
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
                padding=[(padding, padding), (padding, padding)],
                precision=self.precision,
            )
            if self.use_bias:
                shp = [self.features]
                bias = self.param(
                    "bias",
                    self.bias_init,
                    shp,
                    self.param_dtype,
                    channels * self.kernel_size * self.kernel_size,
                    self.features * self.kernel_size * self.kernel_size,
                )
                out = out + bias.reshape(1, 1, 1, -1)

        # reshape to original batch dimensions
        new_height, new_width = out.shape[-3], out.shape[-2]
        out = out.reshape((*batch_dims, new_height, new_width, out.shape[-1]))

        return out


class AttnBlock(nn.Module):
    r"""Self-attention block with group normalization in U-Net models.

    Args:
        num_heads (int): Number of attention heads.
        num_groups (int): Number of groups for `GroupNorm`.
        epsilon (float, optional): Small float added to variance to avoid
            dividing by zero in `GroupNorm`. Default is :math:`1e-5`.
        skip_scale (float, optional): Scaling factor for the residual
            connection output. Default is :math:`1.0`.
        kernel_init (DefaultInitializer, optional): Kernel initializer for the
            dense layers. Default is `default_init(scale=1.0)`.
        bias_init (DefaultInitializer, optional): Bias initializer for the
            dense layers. Default is `default_init(scale=1e-10)`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    num_heads: int
    num_groups: int
    epsilon: float = 1e-5
    skip_scale: float = 1.0
    kernel_init: DefaultInitializer = default_init(scale=1.0)
    bias_init: DefaultInitializer = default_init(scale=1e-10)
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        r"""Forward pass of the `AttnBlock`.

        Args:
            inputs (jax.Array): Input array of shape `(*, H, W, C)`.

        Returns:
            Output array of shape `(*, H, W, C)`.
        """
        batch_dims = inputs.shape[:-3]
        height, width, channels = inputs.shape[-3:]
        inputs = inputs.reshape(-1, height, width, channels)
        chex.assert_rank(inputs, 4)
        inputs = inputs.reshape(inputs.shape[0], height * width, channels)

        norm_in = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm",
        )
        out = norm_in(inputs)

        if self.num_heads == 1:
            # scaled dot-product attention
            qkv_proj = nn.Dense(
                features=3 * channels,
                kernel_init=functools.partial(
                    self.kernel_init,
                    fan_in=channels,
                    fan_out=3 * channels,
                ),
                bias_init=functools.partial(
                    self.bias_init,
                    fan_in=channels,
                    fan_out=3 * channels,
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="qkv_proj",
            )
            qkv = qkv_proj(out)
            query, key, value = jnp.split(qkv, 3, axis=-1)
            scale = 1.0 / jnp.sqrt(jnp.sqrt(channels).astype(self.dtype))
            query = query * scale
            key = key * scale

            attn_weight = jnp.einsum(
                "...qc,...kc->...qk",
                query,
                key,
                precision=self.precision,
            )
            attn_weight = jax.nn.softmax(attn_weight, axis=-1)
            out = jnp.einsum("...qk,...vc->...qc", attn_weight, value)

            out_proj = nn.Dense(
                features=channels,
                kernel_init=functools.partial(
                    self.kernel_init,
                    fan_in=channels,
                    fan_out=channels,
                ),
                bias_init=functools.partial(
                    self.bias_init,
                    fan_in=channels,
                    fan_out=channels,
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="out_proj",
            )
            out = out_proj(out)
        else:
            head_dim = inputs.shape[-1] // self.num_heads
            if head_dim * self.num_heads != inputs.shape[-1]:
                raise ValueError(
                    f"Number of heads {self.num_heads} not compatible with "
                    f"input channels {inputs.shape[-1]}."
                )
            qkv_proj = nn.DenseGeneral(
                features=(self.num_heads, head_dim * 3),
                kernel_init=functools.partial(
                    self.kernel_init,
                    fan_in=channels,
                    fan_out=3 * channels,
                ),
                bias_init=functools.partial(
                    self.bias_init,
                    fan_in=channels,
                    fan_out=3 * channels,
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="qkv_proj",
            )
            qkv = qkv_proj(out)
            query, key, value = jnp.split(qkv, 3, axis=-1)

            scale = 1.0 / jnp.sqrt(jnp.sqrt(head_dim).astype(self.dtype))
            query = query * scale
            key = key * scale
            attn_weight = jnp.einsum(
                "...qhd,...khd->...hqk",
                query,
                key,
                precision=self.precision,
            )
            attn_weight = jax.nn.softmax(attn_weight, axis=-1)
            out = jnp.einsum(
                "...hqk,...khd->...qhd",
                attn_weight,
                value,
                precision=self.precision,
            )
            out_proj = nn.DenseGeneral(
                features=inputs.shape[-1],
                axis=(-2, -1),
                kernel_init=functools.partial(
                    default_init(scale=1e-5),
                    fan_in=channels,
                    fan_out=channels,
                ),
                bias_init=functools.partial(
                    default_init(scale=1e-5),
                    fan_in=channels,
                    fan_out=channels,
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="out_proj",
            )
            out = out_proj(out)

        chex.assert_equal_shape([out, inputs])
        out = out + inputs
        out = out * self.skip_scale
        out = out.reshape((*batch_dims, height, width, channels))

        return out


class EDMUNetBlock(nn.Module):
    r"""U-Net block used in the EDM model.

    .. note::

        This module is adapted from the official PyTorch implementation of EDM:
        `https://github.com/NVlabs/edm/blob/main/training/networks.py`.

    Args:
        features (int): Number of output features.
        resample_filter (typing.Sequence[typing.Union[float, int]], optional):
            One-dimensional FIR filter for resampling. Default is `[1, 1]`.
        downsampling (bool, optional): If `True`, applies a `scale=2`
            downsampling at the beginning of the block. Default is `False`.
        upsampling (bool, optional): If `True`, applies a `scale=2`
            upsampling at the beginning of the block. Default is `False`.
        adaptive_scale (bool, optional): If `True`, uses adaptive scaling
            using the conditioning vector. Default is `False`.
        num_groups (int, optional): Number of groups for `GroupNorm`.
            Default is `32`.
        epsilon (float, optional): Small float added to variance to avoid
            dividing by zero in `GroupNorm`. Default is :math:`1e-5`.
        skip_proj (bool, optional): If `True`, uses a `1x1` convolution for
            the skip connection when the number of input and output channels
            differ. Default is `False`.
        skip_scale (float, optional): Scaling factor for the residual
            connection output. Default is :math:`1.0`.
        num_heads (Optional[int], optional): Number of attention heads.
            If specified, includes an attention block at the end of the U-Net
            block. Default is `None`.
        kernel_init_conv (DefaultInitializer, optional): Kernel initializer for
            the convolutional layers. Default is `default_init()`.
        bias_init_conv (DefaultInitializer, optional): Bias initializer for the
            convolutional layers. Default is `default_init()`.
        kernel_init_attn (DefaultInitializer, optional): Kernel initializer for
            the attention dense layers. Default is `default_init()`.
        bias_init_attn (DefaultInitializer, optional): Bias initializer for the
            attention dense layers. Default is `default_init()`.
        deterministic (Optional[bool], optional): If `True`, disables dropout.
            Default is `None`.
        dropout_rate (float, optional): Dropout rate applied after the
            conditioning integration. Default is :math:`0.0`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
    """

    features: int
    resample_filter: typing.Sequence[typing.Union[float, int]] = (1, 1)
    downsampling: bool = False
    upsampling: bool = False
    adaptive_scale: bool = False
    num_groups: int = 32
    epsilon: float = 1e-5
    skip_proj: bool = False
    skip_scale: float = 1.0
    num_heads: typing.Optional[int] = None
    kernel_init_conv: DefaultInitializer = default_init()
    bias_init_conv: DefaultInitializer = default_init()
    kernel_init_attn: DefaultInitializer = default_init()
    bias_init_attn: DefaultInitializer = default_init()
    deterministic: typing.Optional[bool] = None
    dropout_rate: float = 0.0
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        cond: jax.Array,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass of the `EDMUNetBlock`.

        Args:
            inputs (jax.Array): Input feature map of shape `(*, Hin, Win, Cin)`.
            cond (jax.Array): Conditioning tensor of shape `(*, D)`.
            deterministic (Optional[bool], optional): Whether to apply dropout.
                It merges with the module-level `deterministic` attribute.
                Default is `None`.

        Returns:
            Output feature map of shape `(*, Ho, Wo, C_out)`.
        """
        norm_in = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            name="norm0",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        conv_1 = Conv2D(
            features=self.features,
            kernel_size=3,
            downsampling=self.downsampling,
            upsampling=self.upsampling,
            resample_filter=self.resample_filter,
            kernel_init=self.kernel_init_conv,
            bias_init=self.bias_init_conv,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        out = conv_1(jax.nn.silu(norm_in(inputs.astype(self.dtype))))

        # integrate conditioning
        feat = self.features * 2 if self.adaptive_scale else self.features
        affine = nn.Dense(
            features=feat,
            kernel_init=functools.partial(
                self.kernel_init_conv,
                fan_in=cond.shape[-1],
                fan_out=feat,
            ),
            bias_init=functools.partial(
                self.bias_init_conv,
                fan_in=cond.shape[-1],
                fan_out=feat,
            ),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="cond_emb",
        )
        norm_cond = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            name="norm1",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        emb = affine(cond)[..., None, None, :].astype(self.dtype)
        if self.adaptive_scale:
            scale, shift = jnp.split(emb, 2, axis=-1)
            out = jax.nn.silu(shift + norm_cond(out) * (1 + scale))
        else:
            out = jax.nn.silu(norm_cond(out + emb))

        # residual convolution operation
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )
        dropout = nn.Dropout(rate=self.dropout_rate, name="dropout")
        conv_out = Conv2D(
            features=self.features,
            kernel_size=3,
            kernel_init=self.kernel_init_conv,
            bias_init=self.bias_init_conv,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="conv1",
        )
        out = conv_out(dropout(out, deterministic=m_deterministic))
        if (
            inputs.shape[-1] != self.features
            or self.downsampling
            or self.upsampling
        ):
            conv_skip = Conv2D(
                features=self.features,
                kernel_size=(
                    1
                    if self.skip_proj or inputs.shape[-1] != self.features
                    else None
                ),
                downsampling=self.downsampling,
                upsampling=self.upsampling,
                resample_filter=self.resample_filter,
                kernel_init=self.kernel_init_conv,
                bias_init=self.bias_init_conv,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name="skip_conv",
            )
            out = out + conv_skip(inputs.astype(self.dtype))
        else:
            out = out + inputs.astype(self.dtype)
        out = out * self.skip_scale

        if self.num_heads is not None:
            attn_block = AttnBlock(
                num_heads=self.num_heads,
                num_groups=self.num_groups,
                epsilon=self.epsilon,
                skip_scale=self.skip_scale,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name="attn",
            )
            out = attn_block(out)

        return out


# ==============================================================================
# Networks
class SongNetwork(nn.Module):
    r"""Reimplementation of the DDPM++ and NCSN++ score network.

    .. note::

        This implementation is adapted from the reimplementation in the official
        EDM PyTorch codebase:
        `https://github.com/NVlabs/edm/blob/main/training/networks.py`.

    Args:
    """

    features: int
    channel_mult: typing.Sequence[int] = (1, 2, 2, 2)
    resample_filter: typing.Sequence[typing.Union[float, int]] = (1, 1)
    deterministic: typing.Optional[bool] = None
    dropout_rate: float = 0.10
    num_blocks: int = 4
    attn_resolutions: typing.Sequence[int] = (16,)
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        cond: jax.Array,
        deterministic: bool = False,
    ) -> jax.Array:
        r"""Forward pass of the `SongNetwork` with image and conditioning.

        Args:
            inputs (jax.Array): Input array.
            cond (jax.Array): Conditioning array.
            deterministic (bool, optional): Whether to apply dropout.
                It merges with the module-level `deterministic` attribute.
                Default is `False`.

        Returns:
            Output array.
        """
        height, width = inputs.shape[-3], inputs.shape[-2]
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )
        _block_kwargs: typing.Dict[str, typing.Any] = dict(
            dropout_rate=self.dropout_rate,
            skip_scale=math.sqrt(0.5),
            resample_filter=self.resample_filter,
            epsilon=1e-6,
            skip_proj=True,
            adaptive_scale=False,
            kernel_init_conv=default_init(),
            bias_init_conv=default_init(),
            kernel_init_attn=default_init(scale=math.sqrt(0.2)),
            bias_init_attn=default_init(scale=math.sqrt(0.2)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

        # forward pass the downsampling encoder blocks
        skips = []
        out = inputs.astype(self.dtype)
        for level, mult in enumerate(self.channel_mult):
            h, w = height >> level, width >> level
            if level == 0:
                conv = Conv2D(
                    features=self.features,
                    kernel_size=3,
                    kernel_init=default_init(),
                    bias_init=default_init(),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"enc_{h}x{w}_conv",
                )
                out = conv(out)
            else:
                conv = EDMUNetBlock(
                    features=out.shape[-1],
                    downsampling=True,
                    name=f"enc_{h}x{w}_block_down",
                    **_block_kwargs,
                )
                out = conv(out, cond, deterministic=m_deterministic)
            skips.append(out)

            for idx in range(self.num_blocks):
                block = EDMUNetBlock(
                    features=self.features * mult,
                    num_heads=(
                        1
                        if (
                            h in self.attn_resolutions
                            and w in self.attn_resolutions
                        )
                        else None
                    ),
                    name=f"enc_{h}x{w}_block_{idx}",
                    **_block_kwargs,
                )
                out = block(out, cond, deterministic=m_deterministic)
                skips.append(out)

        # forward pass the decoder upsampling blocks
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            h, w = height >> level, width >> level
            if level == len(self.channel_mult) - 1:
                # NOTE: forward pass the middle bottleneck blocks
                conv_in = EDMUNetBlock(
                    features=out.shape[-1],
                    num_heads=1,
                    name=f"dec_{h}x{w}_in_0",
                    **_block_kwargs,
                )
                out = conv_in(out, cond, deterministic=m_deterministic)
                conv_out = EDMUNetBlock(
                    features=out.shape[-1],
                    name=f"dec_{h}x{w}_in_1",
                    **_block_kwargs,
                )
                out = conv_out(out, cond, deterministic=m_deterministic)
            else:
                conv = EDMUNetBlock(
                    features=out.shape[-1],
                    upsampling=True,
                    name=f"dec_{h}x{w}_block_up",
                    **_block_kwargs,
                )
                out = conv(out, cond, deterministic=m_deterministic)

            for idx in range(self.num_blocks + 1):
                skip = skips.pop()
                out = jnp.concatenate([out, skip], axis=-1)
                block = EDMUNetBlock(
                    features=self.features * mult,
                    num_heads=(
                        1
                        if (
                            h in self.attn_resolutions
                            and w in self.attn_resolutions
                            and idx == self.num_blocks
                        )
                        else None
                    ),
                    name=f"dec_{h}x{w}_block_{idx}",
                    **_block_kwargs,
                )
                out = block(out, cond, deterministic=m_deterministic)

            if level == 0:
                norm_out = nn.GroupNorm(
                    num_groups=self.channel_mult[0],
                    epsilon=1e-6,
                    name=f"dec_{h}x{w}_norm",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
                conv_out = Conv2D(
                    features=inputs.shape[-1],
                    kernel_size=3,
                    kernel_init=default_init(scale=1e-5),
                    bias_init=default_init(scale=1e-5),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"dec_{h}x{w}_conv_out",
                )
                out = conv_out(jax.nn.silu(norm_out(out.astype(self.dtype))))

        return out
