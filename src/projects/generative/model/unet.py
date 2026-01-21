import typing

import chex
from flax import linen as nn
import jax
from jax import numpy as jnp


# ==============================================================================
# Helper Functions
def upfirdn2d(
    inputs: jax.Array,
    kernel: jax.Array,
    scale: int = 2,
    up: bool = False,
) -> jax.Array:
    r"""Perform 2D upsample-filter-downsample on `NHWC` inputs.

    ..note::

        This helper applies a separable finite impulse response (FIR) filter in
        2D while optionally changing spatial resolution. It is conceptually
        similar to the classic *upfirdn* operation:

        * When ``up=True``: the input is upsampled by ``scale`` using zero
        insertion, filtered with the provided kernel, and cropped so that the
        output is ``scale`` times larger in height and width (modulo padding).
        * When ``up=False``: the input is filtered and then downsampled by
        taking every ``scale``-th pixel in each spatial dimension.
        The operation is implemented using ``jax.lax.conv_general_dilated`` with
        per-channel (depthwise) filtering on NHWC tensors.

    Args:
        inputs: Input tensor of shape ``(..., H, W, C)`` where the leading
            dimensions (if any) are treated as batch dimensions, ``H`` and
            ``W`` are spatial dimensions, and ``C`` is the number of channels.
        kernel: One-dimensional FIR kernel used to construct a 2D separable
            filter via outer product. The kernel is normalized inside the
            function, so its absolute scale does not affect the overall gain.
        scale: Integer scaling factor used for upsampling or downsampling.
            Must be a positive integer. When ``up=True``, determines the
            upsampling factor; otherwise, determines the downsampling stride.
        up: If ``True``, perform upsampling followed by filtering. If
            ``False``, perform filtering followed by downsampling.
    Returns:
        Tensor with the same leading batch dimensions and channel
        count as ``inputs``. The spatial dimensions are scaled by ``scale``
        (approximately ``H * scale, W * scale`` when ``up=True``, or
        ``H / scale, W / scale`` when ``up=False``, subject to kernel padding).
    """
    batch_dims = inputs.shape[:-3]
    height, width, channels = inputs.shape[-3:]
    inputs = jnp.reshape(inputs, (-1, height, width, channels))

    k = jnp.array(kernel, dtype=inputs.dtype)
    k = jnp.outer(k, k) / jnp.square(jnp.sum(k))
    k_pad = (k.shape[-1] - 1) // 2

    if up:
        k = jnp.multiply(k, scale**2)
        pad_beg = (k.shape[0] + scale - 2) // 2
        pad_end = (k.shape[0] + scale - 2) - pad_beg
        out = jax.lax.conv_general_dilated(
            lhs=inputs,
            lhs_dilation=(scale, scale),
            rhs=jnp.tile(k[:, :, None, None], (1, 1, 1, channels)),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=channels,
            window_strides=(1, 1),
            padding=[(pad_beg, pad_end), (pad_beg, pad_end)],
        )
    else:
        out = jax.lax.conv_general_dilated(
            lhs=inputs,
            rhs=jnp.tile(k[:, :, None, None], (1, 1, 1, channels)),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=channels,
            window_strides=(scale, scale),
            padding=[(k_pad, k_pad), (k_pad, k_pad)],
        )
    new_height, new_width = out.shape[-3], out.shape[-2]

    out = jnp.reshape(out, (*batch_dims, new_height, new_width, channels))

    return out


# ==============================================================================
# Modules
class ResNetBlock(nn.Module):
    r"""A residual downsampling block with two convolutional layers.

    Args:
        features (int): Dimensionality of the latent features.
        num_groups (int, optional): Number of groups for `GroupNorm`.
            Default is :math:`32`.
        epsilon (float, optional): Small float added to variance to avoid
            dividing by zero in `GroupNorm`. Default is :math:`1e-5`.
        deterministic (bool, optional): If true, the model is run in
            deterministic mode (e.g., no dropout). Defaults to `None`.
        dropout_rate (float, optional): Dropout rate. Default is :math:`0`.
        skip_scale (float, optional): Scaling factor for the residual
            connection output. Default is :math:`1.0`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    features: int
    num_groups: int = 32
    epsilon: float = 1e-5
    deterministic: typing.Optional[bool] = None
    dropout_rate: float = 0.0
    skip_scale: float = 1.0
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    def setup(self) -> None:
        r"""Instantiates a `ResNetBlock` instance."""
        self.norm_1 = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm0",
        )
        self.conv_1 = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv0",
        )
        self.cond_linear = nn.Dense(
            features=self.features,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="cond_in",
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate, name="dropout")

        self.norm_2 = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm1",
        )
        self.conv_2 = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1e-10,
                mode="fan_avg",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv1",
        )

        self.conv_shortcut = nn.Dense(
            features=self.features,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_shortcut",
        )

    def __call__(
        self,
        inputs: jax.Array,
        cond: typing.Optional[jax.Array] = None,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass of the `ResNetBlock`.

        Args:
            inputs (jax.Array): Input array of shape `(*, H, W, C_in)`.
            cond (Optional[jax.Array], optional): Optional conditioning array
                of shape `(*, C_cond)`.
            deterministic (bool, optional): If true, the model is run in
                deterministic mode (e.g., no dropout). Defaults to `None`.

        Returns:
            Output array of shape `(*, H, W, C_out)`, where `C_out` is the
                `features` specified during instantiation.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )
        batch_dims = inputs.shape[:-3]
        dims = chex.Dimensions(
            H=inputs.shape[-3],
            W=inputs.shape[-2],
            C=inputs.shape[-1],
        )

        out = self.conv_1(jax.nn.silu(self.norm_1(inputs)))

        if cond is not None:
            out = out + self.cond_linear(cond)[..., None, None, :]
        out = jax.nn.silu(self.norm_2(out))
        out = self.dropout(out, deterministic=m_deterministic)
        out = self.conv_2(out)

        if inputs.shape[-1] != self.features:
            shortcut = self.conv_shortcut(inputs)
        else:
            shortcut = inputs
        out = out + shortcut
        out = out * self.skip_scale
        chex.assert_shape(out, (*batch_dims, *dims["HW"], self.features))

        return out


class DownsampleBlock(nn.Module):
    r"""A downsampling block using averaging pooling or strided convolution.

    Args:
        with_conv (bool, optional): If true, uses a strided convolution for
            downsampling. If `False`, uses average pooling. Default is `True`.
        features (int, optional): Number of output features. If `None`,
            the number of input features is used. Default is `None`.
        kernel_size (int, optional): Size of the convolutional kernel.
            Default is `3`.
        resample_filter (jax.Array, optional): One-dimensional FIR filter for
            resampling. If `None`, no filtering is applied. Default is `None`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
    """

    with_conv: bool = True
    features: typing.Optional[int] = None
    kernel_size: int = 3
    resample_filter: typing.Optional[jax.Array] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        r"""Forward pass of the `DownsampleBlock`.

        Args:
            inputs (jax.Array): Input array of shape `(*, H, W, C)`.

        Returns:
            Output array of shape `(*, H / 2, W / 2, C)`.
        """
        batch_dims = inputs.shape[:-3]
        dims = chex.Dimensions(
            H=inputs.shape[-3],
            h=inputs.shape[-3] // 2,
            W=inputs.shape[-2],
            w=inputs.shape[-2] // 2,
            C=inputs.shape[-1],
        )
        padding = self.kernel_size // 2

        if self.resample_filter is None:
            if self.with_conv:
                out = nn.Conv(
                    features=(
                        self.features
                        if self.features is not None
                        else inputs.shape[-1]
                    ),
                    kernel_size=(self.kernel_size, self.kernel_size),
                    strides=(2, 2),
                    padding=(padding, padding),
                    kernel_init=jax.nn.initializers.variance_scaling(
                        scale=1.0,
                        mode="fan_avg",
                        distribution="uniform",
                    ),
                    bias_init=jax.nn.initializers.zeros,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name="conv",
                )(inputs)
            else:
                out = nn.avg_pool(inputs, window_shape=(2, 2), strides=(2, 2))
        else:
            out = upfirdn2d(
                inputs,
                kernel=self.resample_filter,
                scale=2,
                up=False,
            )
            out = nn.Conv(
                features=(
                    self.features
                    if self.features is not None
                    else inputs.shape[-1]
                ),
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding=(padding, padding),
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="conv",
            )(out)
        chex.assert_shape(out, (*batch_dims, *dims["hw"], out.shape[-1]))

        return out


class UpsampleBlock(nn.Module):
    r"""An upsampling block using nearest-neighbor interpolation.

    Args:
        antialias (bool, optional): If `True`, applies anti-aliasing when
            calling `jax.image.resize`. Default is `False`.
        method (jax.image.ResizeMethod, optional): The upsampling method to use.
            Default is `jax.image.ResizeMethod.NEAREST`.
        with_conv (bool, optional): If true, applies a convolution after
            upsampling. Default is `True`.
        features (int, optional): Number of output features. If `None`,
            the number of input features is used. Default is `None`.
        kernel_size (int, optional): Size of the convolutional kernel.
            Default is `3`.
        resample_filter (jax.Array, optional): One-dimensional FIR filter for
            resampling. If `None`, no filtering is applied. Default is `None`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    antialias: bool = False
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.NEAREST
    with_conv: bool = True
    features: typing.Optional[int] = None
    kernel_size: int = 3
    resample_filter: typing.Optional[jax.Array] = None
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        r"""Forward pass of the `UpsampleBlock`.

        Args:
            inputs (jax.Array): Input array of shape `(*, H, W, C)`

        Returns:
            Output array of shape `(*, H * 2, W * 2, C)`.
        """
        batch_dims = inputs.shape[:-3]
        dims = chex.Dimensions(
            H=inputs.shape[-3],
            h=inputs.shape[-3] * 2,
            W=inputs.shape[-2],
            w=inputs.shape[-2] * 2,
            C=inputs.shape[-1],
        )

        if self.resample_filter is None:
            out = jax.image.resize(
                inputs,
                shape=(*batch_dims, *dims["hwC"]),
                method=self.method,
                antialias=self.antialias,
                precision=self.precision,
            )
            if self.with_conv:
                padding = self.kernel_size // 2
                out = nn.Conv(
                    features=(
                        self.features
                        if self.features is not None
                        else inputs.shape[-1]
                    ),
                    kernel_size=(self.kernel_size, self.kernel_size),
                    strides=(1, 1),
                    padding=(padding, padding),
                    kernel_init=jax.nn.initializers.variance_scaling(
                        scale=1.0,
                        mode="fan_avg",
                        distribution="uniform",
                    ),
                    bias_init=jax.nn.initializers.zeros,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name="conv",
                )(out)
        else:
            out = upfirdn2d(
                inputs,
                kernel=self.resample_filter,
                scale=2,
                up=True,
            )
            padding = self.kernel_size // 2
            out = nn.Conv(
                features=(
                    self.features
                    if self.features is not None
                    else inputs.shape[-1]
                ),
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding=(padding, padding),
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="conv",
            )(out)

        chex.assert_shape(out, (*batch_dims, *dims["hw"], out.shape[-1]))
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
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    num_heads: int
    num_groups: int
    epsilon: float = 1e-5
    skip_scale: float = 1.0
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
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=0.2, mode="fan_avg", distribution="uniform"
                ),
                bias_init=jax.nn.initializers.zeros,
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


class SongNetBlock(nn.Module):
    r"""A residual block with upsampling/downsampling used in ScoreNet.

    Args:
        features (int): Dimensionality of the latent features.
        num_groups (int, optional): Number of groups for `GroupNorm`.
            Default is :math:`32`.
        epsilon (float, optional): Small float added to variance to avoid
            dividing by zero in `GroupNorm`. Default is :math:`1e-5`.
        deterministic (bool, optional): If true, the model is run in
            deterministic mode (e.g., no dropout). Defaults to `None`.
        dropout_rate (float, optional): Dropout rate. Default is :math:`0`.
        skip_scale (float, optional): Scaling factor for the residual
            connection output. Default is :math:`1.0`.
        resample_filter (Sequence[int], optional): One-dimensional FIR filter
            for resampling. Default is :math:`[1, 1]`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    features: int
    num_groups: int = 32
    epsilon: float = 1e-5
    deterministic: typing.Optional[bool] = None
    dropout_rate: float = 0.0
    skip_scale: float = 1.0
    resample_filter: typing.Sequence[int] = (1, 1)
    upsampling: bool = False
    dtype: typing.Any = None
    param_dtype: typing.Any = None
    precision: typing.Any = None

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        cond: typing.Optional[jax.Array] = None,
        deterministic: typing.Optional[bool] = None,
    ) -> jax.Array:
        r"""Forward pass of the `ResNetBlock`.

        Args:
            inputs (jax.Array): Input array of shape `(*, H, W, C_in)`.
            cond (Optional[jax.Array], optional): Optional conditioning array
                of shape `(*, C_cond)`.
            deterministic (bool, optional): If true, the model is run in
                deterministic mode (e.g., no dropout). Defaults to `None`.

        Returns:
            Output array of shape `(*, H, W, C_out)`, where `C_out` is the
                `features` specified during instantiation.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )

        norm_1 = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm0",
        )
        if self.upsampling:
            conv_1 = UpsampleBlock(
                with_conv=True,
                features=self.features,
                resample_filter=jnp.array(
                    self.resample_filter,
                    dtype=self.param_dtype,
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="upsample",
            )

        else:
            conv_1 = DownsampleBlock(
                with_conv=True,
                features=self.features,
                resample_filter=jnp.array(
                    self.resample_filter,
                    dtype=self.param_dtype,
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="downsample",
            )
        out = conv_1(jax.nn.silu(norm_1(inputs)))

        if cond is not None:
            cond_linear = nn.Dense(
                features=self.features,
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="cond_in",
            )
            out = out + cond_linear(cond)[..., None, None, :]
        dropout = nn.Dropout(rate=self.dropout_rate, name="dropout")
        norm_2 = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm1",
        )
        conv_2 = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1e-10,
                mode="fan_avg",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv1",
        )
        out = jax.nn.silu(norm_2(out))
        out = dropout(out, deterministic=m_deterministic)
        out = conv_2(out)

        if inputs.shape[-3:] != out.shape[-3:]:
            if self.upsampling:
                conv_shortcut = UpsampleBlock(
                    with_conv=inputs.shape[-1] != self.features,
                    features=self.features,
                    kernel_size=1,
                    resample_filter=jnp.array(
                        self.resample_filter,
                        dtype=self.dtype,
                    ),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name="shortcut",
                )
            else:
                conv_shortcut = DownsampleBlock(
                    with_conv=inputs.shape[-1] != self.features,
                    features=self.features,
                    kernel_size=1,
                    resample_filter=jnp.array(
                        self.resample_filter,
                        dtype=self.dtype,
                    ),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name="shortcut",
                )
            shortcut = conv_shortcut(inputs)
        else:
            shortcut = inputs
        out = out + shortcut
        out = out * self.skip_scale

        return out


# ==============================================================================
# Network Wrapper
class HoNetwork(nn.Module):
    r"""U-Net architecture for in denoising deep probabilistic models.

    This module is adapted from the original implementation of the U-Net
    architecture from "Denoising Diffusion Probabilistic Models" by
    Jonathan Ho et al. and the original implementation is available at
    `https://github.com/hojonathanho/diffusion`.

    Args:
        features (int): Base number of features for the latent representations.
        out_features (int, optional): Number of output features. If `None`,
            the number of input features is used. Default is `None`.
        ch_mults (typing.Sequence[int], optional): Sequence of multipliers
            for the number of features at each level of the U-Net architecture.
            Default is `(1, 2, 2, 2)`.
        num_groups (int, optional): Number of groups for `GroupNorm`.
            Default is :math:`32`.
        num_res_blocks (int, optional): Number of residual blocks per level.
            Default is :math:`4`.
        attn_resolutions (typing.Sequence[int], optional): Sequence of spatial
            resolutions at which to apply attention. Default is `(16,)`.
        dropout_rate (float, optional): Dropout rate. Default is :math:`0`.
        epsilon (float, optional): Small float added to variance to avoid
            dividing by zero in `GroupNorm`. Default is :math:`1e-6`.
        resample_with_conv (bool, optional): If `True`, uses convolutional
            layers for upsampling and downsampling.
        deterministic (bool, optional): Whether to apply dropout operations.
            Merged with the `deterministic` argument passed to `__call__`.
            Defaults to `None`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    features: int
    out_features: typing.Optional[int] = None
    ch_mults: typing.Sequence[int] = (1, 2, 2, 2)
    num_groups: int = 32
    num_res_blocks: int = 4
    attn_resolutions: typing.Sequence[int] = (16,)
    dropout_rate: float = 0.0
    epsilon: float = 1e-6
    resample_with_conv: bool = True
    deterministic: typing.Optional[bool] = None
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
        r"""Forward pass of the `HoNetwork" architecture.

        Args:
            inputs (jax.Array): Input array of shape `(*, H, W, C_in)`.
            cond (jax.Array): Conditioning array of shape `(*, C_cond)`.
            deterministic (bool, optional): If true, the model is run in
                deterministic mode (e.g., no dropout). Defaults to `None`.

        Returns:
            Output array of shape `(*, H, W, C_out)`, where `C_out` is the
                number of channels specified by `out_features` during
                instantiation or the number of input channels if
                `out_features` is `None`.
        """
        m_determinisitc = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )
        batch_dims = inputs.shape[:-3]
        dims = chex.Dimensions(
            H=inputs.shape[-3],
            W=inputs.shape[-2],
            C=inputs.shape[-1],
        )
        hs = []

        # forward pass the downsampling path
        out = inputs.astype(self.dtype)
        for level, mult in enumerate(self.ch_mults):
            if level == 0:
                res_block = nn.Conv(
                    features=self.features,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="SAME",
                    kernel_init=jax.nn.initializers.variance_scaling(
                        scale=1.0,
                        mode="fan_avg",
                        distribution="uniform",
                    ),
                    use_bias=True,
                    bias_init=jax.nn.initializers.zeros,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name="conv_in",
                )
                out = res_block(out)
            else:
                res_block = DownsampleBlock(
                    with_conv=self.resample_with_conv,
                    features=out.shape[-1],
                    resample_filter=None,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name=f"down_{level:d}_downsample",
                )
                out = res_block(out)
            hs.append(out)

            for i in range(self.num_res_blocks):
                res_block = ResNetBlock(
                    features=self.features * mult,
                    num_groups=self.num_groups,
                    dropout_rate=self.dropout_rate,
                    epsilon=self.epsilon,
                    skip_scale=1.0,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"down_{level:d}_block_{i + 1:d}",
                )
                out = res_block(
                    inputs=out,
                    cond=cond,
                    deterministic=m_determinisitc,
                )
                if out.shape[-3] in self.attn_resolutions:
                    attn_block = AttnBlock(
                        num_heads=1,
                        num_groups=self.num_groups,
                        epsilon=self.epsilon,
                        skip_scale=1.0,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                        precision=self.precision,
                        name=f"down_{level:d}_attn_{i + 1:d}",
                    )
                    out = attn_block(out)
                hs.append(out)

        # forward pass the upsampling path
        for level, mult in reversed(list(enumerate(self.ch_mults))):
            out_ch = self.features * mult
            if level == len(self.ch_mults) - 1:
                # forward pass the middle blocks
                block = ResNetBlock(
                    features=out.shape[-1],
                    num_groups=self.num_groups,
                    dropout_rate=self.dropout_rate,
                    epsilon=self.epsilon,
                    skip_scale=1.0,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name="mid_block_1",
                )
                out = block(out, cond=cond, deterministic=m_determinisitc)
                block = AttnBlock(
                    num_heads=1,
                    num_groups=self.num_groups,
                    epsilon=self.epsilon,
                    skip_scale=1.0,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name="mid_attn",
                )
                out = block(out)
                block = ResNetBlock(
                    features=out.shape[-1],
                    num_groups=self.num_groups,
                    dropout_rate=self.dropout_rate,
                    epsilon=self.epsilon,
                    skip_scale=1.0,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name="mid_block_2",
                )
                out = block(out, cond=cond, deterministic=m_determinisitc)
            else:
                block = UpsampleBlock(
                    with_conv=self.resample_with_conv,
                    features=out.shape[-1],
                    resample_filter=None,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name=f"up_{level:d}_upsample",
                )
                out = block(out)

            for i in range(self.num_res_blocks + 1):
                skip = hs.pop()
                out = jnp.concatenate([out, skip], axis=-1)
                res_block = ResNetBlock(
                    features=out_ch,
                    num_groups=self.num_groups,
                    dropout_rate=self.dropout_rate,
                    epsilon=self.epsilon,
                    skip_scale=1.0,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"up_{level:d}_block_{i + 1:d}",
                )
                out = res_block(
                    inputs=out,
                    cond=cond,
                    deterministic=m_determinisitc,
                )
                if out.shape[-3] in self.attn_resolutions:
                    attn_block = AttnBlock(
                        num_heads=1,
                        num_groups=self.num_groups,
                        epsilon=self.epsilon,
                        skip_scale=1.0,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                        precision=self.precision,
                        name=f"up_{level:d}_attn_{i + 1:d}",
                    )
                    out = attn_block(out)

        norm_out = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm_out",
        )
        conv_out = nn.Conv(
            features=(
                self.out_features
                if isinstance(
                    self.out_features,
                    int,
                )
                else inputs.shape[-1]
            ),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1e-10,
                mode="fan_avg",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_out",
        )
        out = conv_out(jax.nn.silu(norm_out(out)))
        chex.assert_shape(out, (*batch_dims, *dims["HWC"]))

        return out


class SongNetwork(nn.Module):
    r"""U-Net architecture for score-function estimation.

    This module is adapted from the original implementation of the U-Net
    architecture from "Score-Based Generative Modeling through Stochastic
    Differential Equations" by Yang Song et al. and the original implementation
    is available at `https://github.com/yang-song/score_sde_pytorch`.

    Args:
        features (int): Base number of features for the latent representations.
        ch_mults (typing.Sequence[int], optional): Sequence of multipliers
            for the number of features at each level of the U-Net.
        num_groups (int, optional): Number of groups for `GroupNorm`.
        num_res_blocks (int, optional): Number of residual blocks per level.
        attn_resolutions (typing.Sequence[int], optional): Sequence of
            resolutions at which to apply attention mechanisms.
        dropout_rate (float, optional): Dropout rate. Default is :math:`0.0`.
        epsilon (float, optional): Small float added to variance to avoid
            dividing by zero in `GroupNorm`. Default is :math:`1e-6`.
        resample_filter (typing.Sequence[int]): One-dimensional FIR filter
            for resampling. Default is `[1, 1]`.
        skip_scale (float, optional): Scaling factor for the residual
            connection outputs. Default is :math:`1.0`.
        deterministic (bool, optional): If true, the model is run in
            deterministic mode (e.g., no dropout). Defaults to `None`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    features: int
    ch_mults: typing.Sequence[int] = (2, 2, 2)
    num_groups: int = 32
    num_res_blocks: int = 4
    attn_resolutions: typing.Sequence[int] = (16,)
    dropout_rate: float = 0.0
    epsilon: float = 1e-6
    resample_filter: typing.Sequence[int] = (1, 1)
    skip_scale: float = 1.0
    deterministic: typing.Optional[bool] = None
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
        r"""Forward pass of the `ScoreNet`.

        Args:
            inputs (jax.Array): Input array of shape `(*, H, W, C_in)`.
            cond (jax.Array): Conditioning array of shape `(*, C_cond)`.
            deterministic (bool, optional): If true, the model is run in
                deterministic mode (e.g., no dropout). Defaults to `None`.

        Returns:
            Output array of shape `(*, H, W, C_out)`, where `C_out` is the
                number of channels in the input.
        """
        m_deterministic = nn.merge_param(
            "deterministic",
            self.deterministic,
            deterministic,
        )
        batch_dims = inputs.shape[:-3]
        dims = chex.Dimensions(
            H=inputs.shape[-3],
            W=inputs.shape[-2],
            C=inputs.shape[-1],
        )
        skips = []

        # forward pass the downsampling path
        out = inputs.astype(self.dtype)
        for level, mult in enumerate(self.ch_mults):
            if level == 0:
                res_block = nn.Conv(
                    features=self.features,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=(1, 1),
                    kernel_init=jax.nn.initializers.variance_scaling(
                        scale=1.0,
                        mode="fan_avg",
                        distribution="uniform",
                    ),
                    bias_init=jax.nn.initializers.zeros,
                    dtype=self.dtype,
                    name=f"enc_{out.shape[-3]}x{out.shape[-2]}_conv",
                )
                out = res_block(out)
            else:
                res_block = SongNetBlock(
                    features=self.features * mult,
                    num_groups=self.num_groups,
                    dropout_rate=self.dropout_rate,
                    epsilon=self.epsilon,
                    skip_scale=self.skip_scale,
                    resample_filter=self.resample_filter,
                    upsampling=False,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"enc_{out.shape[-3]}x{out.shape[-2]}_down",
                )
                out = res_block(out, cond=cond, deterministic=m_deterministic)
            skips.append(out)

            for i in range(self.num_res_blocks):
                res_block = ResNetBlock(
                    features=self.features * mult,
                    num_groups=self.num_groups,
                    dropout_rate=self.dropout_rate,
                    epsilon=self.epsilon,
                    skip_scale=self.skip_scale,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"enc_{out.shape[-3]}x{out.shape[-2]}_blk_{i + 1:d}",
                )
                out = res_block(
                    inputs=out,
                    cond=cond,
                    deterministic=m_deterministic,
                )
                if out.shape[-3] in self.attn_resolutions:
                    block = AttnBlock(
                        num_heads=1,
                        num_groups=self.num_groups,
                        epsilon=self.epsilon,
                        skip_scale=self.skip_scale,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                        precision=self.precision,
                        name=f"attn_{out.shape[-3]}x{out.shape[-2]}_{i + 1:d}",
                    )
                    out = block(out)
                skips.append(out)

        # forward pass the upsampling path
        for level, mult in reversed(list(enumerate(self.ch_mults))):
            out_ch = self.features * mult
            if level == len(self.ch_mults) - 1:
                # forward pass middle blocks
                block = ResNetBlock(
                    features=out.shape[-1],
                    num_groups=self.num_groups,
                    epsilon=self.epsilon,
                    skip_scale=self.skip_scale,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"dec_{out.shape[-3]}x{out.shape[-2]}_conv_in",
                )
                out = block(out, cond=cond, deterministic=m_deterministic)
                block = AttnBlock(
                    num_heads=1,
                    num_groups=self.num_groups,
                    epsilon=self.epsilon,
                    skip_scale=self.skip_scale,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"dec_{out.shape[-3]}x{out.shape[-2]}_attn",
                )
                out = block(out)
                block = ResNetBlock(
                    features=out.shape[-1],
                    num_groups=self.num_groups,
                    epsilon=self.epsilon,
                    skip_scale=self.skip_scale,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"dec_{out.shape[-3]}x{out.shape[-2]}_conv_out",
                )
                out = block(out, cond=cond, deterministic=m_deterministic)
            else:
                res_block = SongNetBlock(
                    features=out_ch,
                    dropout_rate=self.dropout_rate,
                    num_groups=self.num_groups,
                    epsilon=self.epsilon,
                    skip_scale=self.skip_scale,
                    resample_filter=self.resample_filter,
                    upsampling=True,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"dec_{out.shape[-3]}x{out.shape[-2]}_up",
                )
                out = res_block(out, cond=cond, deterministic=m_deterministic)

            for i in range(self.num_res_blocks + 1):
                skip = skips.pop()
                out = jnp.concatenate([out, skip], axis=-1)
                res_block = ResNetBlock(
                    features=out_ch,
                    num_groups=self.num_groups,
                    dropout_rate=self.dropout_rate,
                    epsilon=self.epsilon,
                    skip_scale=self.skip_scale,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"dec_{out.shape[-3]}x{out.shape[-2]}_blk_{i + 1:d}",
                )
                out = res_block(
                    inputs=out,
                    cond=cond,
                    deterministic=m_deterministic,
                )
                if (
                    out.shape[-3] in self.attn_resolutions
                    and i == self.num_res_blocks
                ):
                    block = AttnBlock(
                        num_heads=1,
                        num_groups=self.num_groups,
                        epsilon=self.epsilon,
                        skip_scale=self.skip_scale,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                        precision=self.precision,
                        name=f"attn_{out.shape[-3]}x{out.shape[-2]}_{i + 1:d}",
                    )
                    out = block(out)

        # forward pass the output convolution
        norm_out = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm_out",
        )
        out = jax.nn.silu(norm_out(out))
        conv_out = nn.Conv(
            features=dims.C,  # type: ignore
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1e-10,
                mode="fan_avg",
                distribution="uniform",
            ),
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            name="conv_out",
        )
        out = conv_out(out)
        chex.assert_shape(out, (*batch_dims, *dims["HWC"]))

        return out
