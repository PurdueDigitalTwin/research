import typing

import chex
from flax import linen as nn
import jax
from jax import numpy as jnp


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
    kernel_init: typing.Callable = jax.nn.initializers.lecun_normal()
    bias_init: typing.Callable = jax.nn.initializers.zeros
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
        kernel_init (Callable, optional): Kernel initializer for the dense
            layers. Default is `jax.nn.initializers.lecun_normal()`.
        bias_init (Callable, optional): Bias initializer for the dense layers.
            Default is `jax.nn.initializers.zeros`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    num_heads: int
    num_groups: int
    epsilon: float = 1e-5
    skip_scale: float = 1.0
    kernel_init: typing.Callable = jax.nn.initializers.lecun_normal()
    bias_init: typing.Callable = jax.nn.initializers.zeros
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
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
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
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
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
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=0.2,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                bias_init=jax.nn.initializers.zeros,
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
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
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
        kernel_init_conv (Callable, optional): Kernel initializer for the
            convolutional layers. Default is `jax.nn.initializers.lecun_normal()`.
        bias_init_conv (Callable, optional): Bias initializer for the
            convolutional layers. Default is `jax.nn.initializers.zeros`.
        kernel_init_attn (Callable, optional): Kernel initializer for the
            attention dense layers. Default is
            `jax.nn.initializers.lecun_normal()`.
        bias_init_attn (Callable, optional): Bias initializer for the
            attention dense layers. Default is `jax.nn.initializers.zeros`.
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
    kernel_init_conv: typing.Callable = jax.nn.initializers.lecun_normal()
    bias_init_conv: typing.Callable = jax.nn.initializers.zeros
    kernel_init_attn: typing.Callable = jax.nn.initializers.lecun_normal()
    bias_init_attn: typing.Callable = jax.nn.initializers.zeros
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
            deterministic (Optional[bool], optional): Whether to

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
        affine = nn.Dense(
            features=(
                self.features * 2 if self.adaptive_scale else self.features
            ),
            kernel_init=self.kernel_init_conv,
            bias_init=self.bias_init_conv,
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
