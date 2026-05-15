"""Stable Diffusion Variational Autoencoder (AutoencoderKL) in Flax.

Implements the KL-regularized autoencoder architecture from Stable
Diffusion, with pretrained weight loading. No ``diffusers`` dependency.

Architecture overview::

    Encoder: Conv -> DownBlocks -> MidBlock -> Norm -> Conv
    Decoder: Conv -> MidBlock -> UpBlocks -> Norm -> Conv
    AutoencoderKL: Encoder -> quant_conv | post_quant_conv -> Decoder

Weight-compatible with ``pcuenq/sd-vae-ft-mse-flax``. Download
weights with::

    huggingface-cli download pcuenq/sd-vae-ft-mse-flax \\
        --local-dir /path/to/sd-vae-ft-mse-flax
"""

import json
import math
import os
import typing

import flax.linen as nn
import flax.serialization
import huggingface_hub
import jax
import jax.numpy as jnp

#: Default latent scaling factor for Stable Diffusion VAE.
SD_VAE_SCALING_FACTOR = 0.18215

#: Default SD VAE channel configuration.
SD_VAE_BLOCK_OUT_CHANNELS = (128, 256, 512, 512)


# ================================================================
# Building blocks
# ================================================================


class ResNetBlock(nn.Module):
    r"""ResNet block with GroupNorm and SiLU activation.

    Attributes:
        out_channels (int): Number of output channels.
        norm_num_groups (int): Groups for GroupNorm.
        dtype (Any): Computation dtype.
        param_dtype (Any): Parameter dtype.
    """

    out_channels: int
    norm_num_groups: int = 32
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x (jax.Array): Input tensor ``(B, H, W, C)``.

        Returns:
            Output tensor ``(B, H, W, out_channels)``.
        """
        in_ch = x.shape[-1]
        residual = x

        h = nn.GroupNorm(
            num_groups=self.norm_num_groups,
            epsilon=1e-6,
            name="norm1",
        )(x)
        h = nn.silu(h)
        h = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv1",
        )(h)

        h = nn.GroupNorm(
            num_groups=self.norm_num_groups,
            epsilon=1e-6,
            name="norm2",
        )(h)
        h = nn.silu(h)
        h = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv2",
        )(h)

        if in_ch != self.out_channels:
            residual = nn.Conv(
                self.out_channels,
                kernel_size=(1, 1),
                padding="VALID",
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="conv_shortcut",
            )(residual)

        return h + residual


class AttentionBlock(nn.Module):
    r"""Single-head self-attention block with GroupNorm.

    Attributes:
        channels (int): Number of channels.
        norm_num_groups (int): Groups for GroupNorm.
        dtype (Any): Computation dtype.
        param_dtype (Any): Parameter dtype.
    """

    channels: int
    norm_num_groups: int = 32
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x (jax.Array): Input tensor ``(B, H, W, C)``.

        Returns:
            Output tensor ``(B, H, W, C)``.
        """
        b, h, w, c = x.shape
        residual = x

        y = nn.GroupNorm(
            num_groups=self.norm_num_groups,
            epsilon=1e-6,
            name="group_norm",
        )(x)
        y = y.reshape(b, h * w, c)

        q = nn.Dense(
            c,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="query",
        )(y)
        k = nn.Dense(
            c,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="key",
        )(y)
        v = nn.Dense(
            c,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="value",
        )(y)

        scale = 1.0 / math.sqrt(c)
        attn = jnp.einsum("bic,bjc->bij", q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        y = jnp.einsum("bij,bjc->bic", attn, v)

        y = nn.Dense(
            c,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="proj_attn",
        )(y)
        y = y.reshape(b, h, w, c)

        return y + residual


class Downsample(nn.Module):
    r"""Stride-2 convolution for spatial downsampling.

    Uses asymmetric padding ``((0,1),(0,1))`` to match the
    Stable Diffusion convention.

    Attributes:
        out_channels (int): Number of output channels.
        dtype (Any): Computation dtype.
        param_dtype (Any): Parameter dtype.
    """

    out_channels: int
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Downsample by 2x.

        Args:
            x (jax.Array): Input ``(B, H, W, C)``.

        Returns:
            Output ``(B, H/2, W/2, out_channels)``.
        """
        return nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((0, 1), (0, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv",
        )(x)


class Upsample(nn.Module):
    r"""Nearest-neighbor upsample + convolution.

    Attributes:
        out_channels (int): Number of output channels.
        dtype (Any): Computation dtype.
        param_dtype (Any): Parameter dtype.
    """

    out_channels: int
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Upsample by 2x.

        Args:
            x (jax.Array): Input ``(B, H, W, C)``.

        Returns:
            Output ``(B, 2H, 2W, out_channels)``.
        """
        b, h, w, c = x.shape
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
        return nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv",
        )(x)


# ================================================================
# Composite blocks
# ================================================================


class DownEncoderBlock(nn.Module):
    r"""Encoder block: ResNet layers + optional downsample.

    Attributes:
        out_channels (int): Output channels for each ResNet.
        num_layers (int): Number of ResNet blocks.
        add_downsample (bool): Whether to add stride-2 downsample.
        norm_num_groups (int): Groups for GroupNorm.
        dtype (Any): Computation dtype.
        param_dtype (Any): Parameter dtype.
    """

    out_channels: int
    num_layers: int = 2
    add_downsample: bool = True
    norm_num_groups: int = 32
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x (jax.Array): Input tensor ``(B, H, W, C)``.

        Returns:
            Output tensor, spatially halved if downsampled.
        """
        for i in range(self.num_layers):
            x = ResNetBlock(
                out_channels=self.out_channels,
                norm_num_groups=self.norm_num_groups,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"resnets_{i}",
            )(x)
        if self.add_downsample:
            x = Downsample(
                out_channels=self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="downsamplers_0",
            )(x)
        return x


class UpDecoderBlock(nn.Module):
    r"""Decoder block: ResNet layers + optional upsample.

    Attributes:
        out_channels (int): Output channels for each ResNet.
        num_layers (int): Number of ResNet blocks.
        add_upsample (bool): Whether to add 2x upsample.
        norm_num_groups (int): Groups for GroupNorm.
        dtype (Any): Computation dtype.
        param_dtype (Any): Parameter dtype.
    """

    out_channels: int
    num_layers: int = 3
    add_upsample: bool = True
    norm_num_groups: int = 32
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x (jax.Array): Input tensor ``(B, H, W, C)``.

        Returns:
            Output tensor, spatially doubled if upsampled.
        """
        for i in range(self.num_layers):
            x = ResNetBlock(
                out_channels=self.out_channels,
                norm_num_groups=self.norm_num_groups,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"resnets_{i}",
            )(x)
        if self.add_upsample:
            x = Upsample(
                out_channels=self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="upsamplers_0",
            )(x)
        return x


class MidBlock(nn.Module):
    r"""Mid block: ResNet -> Attention -> ResNet.

    Attributes:
        channels (int): Number of channels.
        norm_num_groups (int): Groups for GroupNorm.
        dtype (Any): Computation dtype.
        param_dtype (Any): Parameter dtype.
    """

    channels: int
    norm_num_groups: int = 32
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x (jax.Array): Input tensor ``(B, H, W, C)``.

        Returns:
            Output tensor ``(B, H, W, C)``.
        """
        x = ResNetBlock(
            out_channels=self.channels,
            norm_num_groups=self.norm_num_groups,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="resnets_0",
        )(x)
        x = AttentionBlock(
            channels=self.channels,
            norm_num_groups=self.norm_num_groups,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="attentions_0",
        )(x)
        x = ResNetBlock(
            out_channels=self.channels,
            norm_num_groups=self.norm_num_groups,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="resnets_1",
        )(x)
        return x


# ================================================================
# Encoder / Decoder
# ================================================================


class Encoder(nn.Module):
    r"""VAE encoder: image -> latent moments.

    Outputs ``2 * latent_channels`` channels (mean and logvar
    concatenated along the channel axis).

    Attributes:
        latent_channels (int): Number of latent channels.
        block_out_channels (Tuple[int, ...]): Per-block channels.
        layers_per_block (int): ResNets per encoder block.
        norm_num_groups (int): Groups for GroupNorm.
        dtype (Any): Computation dtype.
        param_dtype (Any): Parameter dtype.
    """

    latent_channels: int = 4
    block_out_channels: typing.Tuple[int, ...] = (
        128,
        256,
        512,
        512,
    )
    layers_per_block: int = 2
    norm_num_groups: int = 32
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode input images.

        Args:
            x (jax.Array): Input images ``(B, H, W, C_in)``.

        Returns:
            Moments tensor ``(B, H', W', 2 * latent_channels)``.
        """
        x = nn.Conv(
            self.block_out_channels[0],
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_in",
        )(x)

        num_blocks = len(self.block_out_channels)
        for i in range(num_blocks):
            is_final = i == num_blocks - 1
            x = DownEncoderBlock(
                out_channels=self.block_out_channels[i],
                num_layers=self.layers_per_block,
                add_downsample=not is_final,
                norm_num_groups=self.norm_num_groups,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"down_blocks_{i}",
            )(x)

        x = MidBlock(
            channels=self.block_out_channels[-1],
            norm_num_groups=self.norm_num_groups,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mid_block",
        )(x)

        x = nn.GroupNorm(
            num_groups=self.norm_num_groups,
            epsilon=1e-6,
            name="conv_norm_out",
        )(x)
        x = nn.silu(x)
        x = nn.Conv(
            2 * self.latent_channels,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_out",
        )(x)

        return x


class Decoder(nn.Module):
    r"""VAE decoder: latent -> image.

    Attributes:
        out_channels (int): Number of output image channels.
        block_out_channels (Tuple[int, ...]): Per-block channels
            (same order as encoder; reversed internally).
        layers_per_block (int): Base ResNet count (decoder uses
            ``layers_per_block + 1`` per block).
        norm_num_groups (int): Groups for GroupNorm.
        dtype (Any): Computation dtype.
        param_dtype (Any): Parameter dtype.
    """

    out_channels: int = 3
    block_out_channels: typing.Tuple[int, ...] = (
        128,
        256,
        512,
        512,
    )
    layers_per_block: int = 2
    norm_num_groups: int = 32
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Decode latent representations.

        Args:
            x (jax.Array): Latent tensor ``(B, H', W', C_z)``.

        Returns:
            Reconstructed images ``(B, H, W, C_out)``.
        """
        reversed_channels = list(reversed(self.block_out_channels))

        x = nn.Conv(
            reversed_channels[0],
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_in",
        )(x)

        x = MidBlock(
            channels=reversed_channels[0],
            norm_num_groups=self.norm_num_groups,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mid_block",
        )(x)

        num_blocks = len(reversed_channels)
        for i in range(num_blocks):
            is_final = i == num_blocks - 1
            x = UpDecoderBlock(
                out_channels=reversed_channels[i],
                num_layers=self.layers_per_block + 1,
                add_upsample=not is_final,
                norm_num_groups=self.norm_num_groups,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"up_blocks_{i}",
            )(x)

        x = nn.GroupNorm(
            num_groups=self.norm_num_groups,
            epsilon=1e-6,
            name="conv_norm_out",
        )(x)
        x = nn.silu(x)
        x = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv_out",
        )(x)

        return x


# ================================================================
# AutoencoderKL
# ================================================================


class AutoencoderKL(nn.Module):
    r"""KL-regularized autoencoder for latent diffusion.

    This implements the VAE used in Stable Diffusion. Images are
    encoded to a lower-dimensional latent space and decoded back.
    Pretrained weights can be loaded from HuggingFace via
    ``from_pretrained``.

    Attributes:
        in_channels (int): Input image channels (default 3).
        out_channels (int): Output image channels (default 3).
        latent_channels (int): Latent space channels (default 4).
        block_out_channels (Tuple[int, ...]): Per-block channels.
        layers_per_block (int): ResNets per encoder block.
        norm_num_groups (int): Groups for GroupNorm.
        scaling_factor (float): Latent scaling factor.
        dtype (Any): Computation dtype.
        param_dtype (Any): Parameter dtype.
    """

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: typing.Tuple[int, ...] = (
        128,
        256,
        512,
        512,
    )
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = SD_VAE_SCALING_FACTOR
    dtype: typing.Any = None
    param_dtype: typing.Any = None

    def setup(self):
        """Initialize encoder, decoder, and quant convolutions."""
        self.encoder = Encoder(
            latent_channels=self.latent_channels,
            block_out_channels=self.block_out_channels,
            layers_per_block=self.layers_per_block,
            norm_num_groups=self.norm_num_groups,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.decoder = Decoder(
            out_channels=self.out_channels,
            block_out_channels=self.block_out_channels,
            layers_per_block=self.layers_per_block,
            norm_num_groups=self.norm_num_groups,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.quant_conv = nn.Conv(
            2 * self.latent_channels,
            kernel_size=(1, 1),
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.post_quant_conv = nn.Conv(
            self.latent_channels,
            kernel_size=(1, 1),
            padding="VALID",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Full encode-decode round trip (for init / testing).

        Args:
            x (jax.Array): Input images ``(B, H, W, C_in)``.

        Returns:
            Reconstructed images ``(B, H, W, C_out)``.
        """
        mean, _ = self.encode(x)
        return self.decode(mean)

    def encode(self, x: jax.Array) -> typing.Tuple[jax.Array, jax.Array]:
        """Encode images to latent distribution parameters.

        Args:
            x (jax.Array): Input images ``(B, H, W, C_in)``,
                normalized to ``[-1, 1]``.

        Returns:
            Tuple of ``(mean, logvar)``, each with shape
            ``(B, H', W', latent_channels)``.
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = jnp.split(moments, 2, axis=-1)
        logvar = jnp.clip(logvar, -30.0, 20.0)
        return mean, logvar

    def decode(self, z: jax.Array) -> jax.Array:
        """Decode latent representations to images.

        Args:
            z (jax.Array): Latent tensor
                ``(B, H', W', latent_channels)``.

        Returns:
            Reconstructed images ``(B, H, W, C_out)``.
        """
        z = self.post_quant_conv(z)
        return self.decoder(z)

    @staticmethod
    def from_pretrained(
        path: str,
        dtype: typing.Any = None,
        param_dtype: typing.Any = None,
    ) -> typing.Tuple["AutoencoderKL", typing.Dict]:
        """Load pretrained AutoencoderKL.

        Accepts either a local directory or a HuggingFace repo ID
        (e.g. ``pcuenq/sd-vae-ft-mse-flax``).  When ``path`` is not
        an existing local directory the weights are downloaded from
        HuggingFace Hub automatically.

        Args:
            path (str): Local directory or HuggingFace repo ID
                containing ``config.json`` and
                ``diffusion_flax_model.msgpack``.
            dtype (Any): Computation dtype override.
            param_dtype (Any): Parameter dtype override.

        Returns:
            Tuple of ``(model, params)`` where ``params`` is a
            nested dict ready for ``model.apply``.
        """
        if not os.path.isdir(path):
            path = huggingface_hub.snapshot_download(
                repo_id=path,
                revision="7581b0a0489cc8483c876a728b830b9ce087cf03",
                token=os.getenv("HF_TOKEN", None),
            )
        config_file = os.path.join(path, "config.json")
        with open(config_file) as f:
            config = json.load(f)

        model = AutoencoderKL(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 3),
            latent_channels=config.get("latent_channels", 4),
            block_out_channels=tuple(
                config.get(
                    "block_out_channels",
                    [128, 256, 512, 512],
                )
            ),
            layers_per_block=config.get("layers_per_block", 2),
            norm_num_groups=config.get("norm_num_groups", 32),
            scaling_factor=config.get("scaling_factor", SD_VAE_SCALING_FACTOR),
            dtype=dtype,
            param_dtype=param_dtype,
        )

        weights_file = os.path.join(path, "diffusion_flax_model.msgpack")
        with open(weights_file, "rb") as f:
            params = flax.serialization.from_bytes(None, f.read())

        return model, params
