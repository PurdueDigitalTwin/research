import typing

import chex
from flax import linen as nn
import jax
from jax import numpy as jnp


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
                scale=1.0,
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
            out = out + self.cond_linear(jax.nn.silu(cond))[..., None, None, :]
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
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
    """

    with_conv: bool = True
    dtype: typing.Any = None
    param_dtype: typing.Any = None

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

        if self.with_conv:
            out = nn.Conv(
                features=inputs.shape[-1],
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=((0, 1), (0, 1)),
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="conv0",
            )(inputs)
        else:
            out = nn.avg_pool(inputs, window_shape=(2, 2), strides=(2, 2))
        chex.assert_shape(out, (*batch_dims, *dims["hwC"]))

        return out


class UpsampleBlock(nn.Module):
    r"""An upsampling block using nearest-neighbor interpolation.

    Args:
        with_conv (bool, optional): If true, applies a convolution after
            upsampling. Default is `True`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    with_conv: bool = True
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

        out = jax.image.resize(
            inputs,
            shape=(*batch_dims, *dims["hwC"]),
            method="nearest",
            antialias=True,
            precision=self.precision,
        )
        if self.with_conv:
            out = nn.Conv(
                features=inputs.shape[-1],
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
            )(out)

        chex.assert_shape(out, (*batch_dims, *dims["hwC"]))
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
            q_proj = nn.Dense(
                features=inputs.shape[-1],
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="q_proj",
            )
            query = q_proj(out)
            k_proj = nn.Dense(
                features=inputs.shape[-1],
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="k_proj",
            )
            key = k_proj(out)
            v_proj = nn.Dense(
                features=inputs.shape[-1],
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="v_proj",
            )
            value = v_proj(out)
            out = nn.dot_product_attention(
                query[..., None, :],
                key[..., None, :],
                value[..., None, :],
                broadcast_dropout=False,
                dropout_rate=0.0,
                dtype=self.dtype,
                precision=self.precision,
            )
            out_proj = nn.Dense(
                features=inputs.shape[-1],
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="out_proj",
            )
            out = out_proj(out[..., 0, :])
        else:
            head_dim = inputs.shape[-1] // self.num_heads
            if head_dim * self.num_heads != inputs.shape[-1]:
                raise ValueError(
                    f"Number of heads {self.num_heads} not compatible with "
                    f"input channels {inputs.shape[-1]}."
                )
            q_proj = nn.DenseGeneral(
                features=(self.num_heads, head_dim),
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="q_proj",
            )
            query = q_proj(out)
            k_proj = nn.DenseGeneral(
                features=(self.num_heads, head_dim),
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="k_proj",
            )
            key = k_proj(out)
            v_proj = nn.DenseGeneral(
                features=(self.num_heads, head_dim),
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                bias_init=jax.nn.initializers.zeros,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="v_proj",
            )
            value = v_proj(out)
            out = nn.dot_product_attention(
                query,
                key,
                value,
                broadcast_dropout=False,
                dropout_rate=0.0,
                dtype=self.dtype,
                precision=self.precision,
            )
            out_proj = nn.DenseGeneral(
                features=inputs.shape[-1],
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

        return out


class ScoreNet(nn.Module):
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
            dividing by zero in `GroupNorm`. Default is :math:`1e-5`.
        skip_scale (float, optional): Scaling factor for the residual
            connection outputs. Default is :math:`1.0`.
        deterministic (bool, optional): If true, the model is run in
            deterministic mode (e.g., no dropout). Defaults to `None`.
        dtype (Any, optional): The dtype of the computation.
        param_dtype (Any, optional): The dtype of the parameters.
        precision (Any, optional): Numerical precision of the computation.
    """

    features: int
    ch_mults: typing.Sequence[int] = (1, 2, 2, 2)
    num_groups: int = 32
    num_res_blocks: int = 4
    attn_resolutions: typing.Sequence[int] = (16,)
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
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

        # forward pass the input convolution
        conv_in = nn.Conv(
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
            name="conv_in",
        )
        out = conv_in(inputs)
        skips.append(out)

        # forward pass the downsampling path
        for level, mult in enumerate(self.ch_mults):
            out_ch = self.features * mult
            for i in range(self.num_res_blocks):
                res_block = ResNetBlock(
                    features=out_ch,
                    num_groups=self.num_groups,
                    dropout_rate=self.dropout_rate,
                    epsilon=self.epsilon,
                    skip_scale=self.skip_scale,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"down_resnet_{level + 1:d}_{i + 1:d}",
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
                        name=f"down_attn_{level + 1:d}_{i + 1:d}",
                    )
                    out = block(out)
                skips.append(out)
            if level != len(self.ch_mults) - 1:
                downsample = DownsampleBlock(
                    with_conv=True,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name=f"downsample_{level + 1:d}",
                )
                out = downsample(out)
                skips.append(out)

        # forward pass the middle blocks
        block = ResNetBlock(
            features=out.shape[-1],
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            skip_scale=self.skip_scale,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="mid_resnet_1",
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
            name="mid_attn",
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
            name="mid_resnet_2",
        )
        out = block(out, cond=cond, deterministic=m_deterministic)

        # forward pass the upsampling path
        for level, mult in reversed(list(enumerate(self.ch_mults))):
            out_ch = self.features * mult
            for i in range(self.num_res_blocks + 1):
                skip = skips.pop()
                out = jnp.concatenate([out, skip], axis=-1)
                res_block = ResNetBlock(
                    features=out_ch,
                    dropout_rate=self.dropout_rate,
                    num_groups=self.num_groups,
                    epsilon=self.epsilon,
                    skip_scale=self.skip_scale,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"up_resnet_{level + 1:d}_{i + 1:d}",
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
                        name=f"up_attn_{level + 1:d}_{i + 1:d}",
                    )
                    out = block(out)
            if level != 0:
                upsample = UpsampleBlock(
                    with_conv=True,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    name=f"upsample_{level + 1:d}",
                )
                out = upsample(out)

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
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            dtype=self.dtype,
            name="conv_out",
        )
        out = conv_out(out)
        chex.assert_shape(out, (*batch_dims, *dims["HWC"]))

        return out
