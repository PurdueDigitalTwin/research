"""Tests for AutoencoderKL (SD VAE)."""

import jax
import jax.numpy as jnp
import pytest

from src.projects.generative.model import vae


class TestAutoEncoderKL:
    """Tests for the AutoencoderKL module."""

    @pytest.fixture()
    def small_model(self):
        """Small AutoencoderKL for fast tests."""
        return vae.AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
        )

    def test_init(self, small_model):
        """Model initializes without error."""
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 64, 64, 3))
        variables = small_model.init(key, x)
        assert "params" in variables

    def test_encode_shapes(self, small_model):
        """Encode produces correct latent shapes."""
        key = jax.random.PRNGKey(0)
        x = jnp.ones((2, 64, 64, 3))
        params = small_model.init(key, x)["params"]

        mean, logvar = small_model.apply(
            {"params": params}, x, method=small_model.encode
        )
        # 1 downsample -> 64/2 = 32
        assert mean.shape == (2, 32, 32, 4)
        assert logvar.shape == (2, 32, 32, 4)

    def test_decode_shapes(self, small_model):
        """Decode produces correct image shapes."""
        key = jax.random.PRNGKey(0)
        x = jnp.ones((2, 64, 64, 3))
        params = small_model.init(key, x)["params"]

        z = jnp.ones((2, 32, 32, 4))
        decoded = small_model.apply(
            {"params": params}, z, method=small_model.decode
        )
        assert decoded.shape == (2, 64, 64, 3)

    def test_round_trip(self, small_model):
        """Encode then decode preserves spatial dimensions."""
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 64, 64, 3))
        params = small_model.init(key, x)["params"]

        out = small_model.apply({"params": params}, x)
        assert out.shape == x.shape

    def test_logvar_clipping(self, small_model):
        """Logvar is clipped to [-30, 20]."""
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1, 64, 64, 3)) * 100.0
        params = small_model.init(key, x)["params"]

        _, logvar = small_model.apply(
            {"params": params}, x, method=small_model.encode
        )
        assert jnp.all(logvar >= -30.0)
        assert jnp.all(logvar <= 20.0)

    def test_sd_vae_config(self):
        """Full SD VAE config initializes correctly."""
        model = vae.AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            norm_num_groups=32,
        )
        key = jax.random.PRNGKey(0)
        # Use small spatial size to keep test fast.
        x = jnp.ones((1, 32, 32, 3))
        variables = model.init(key, x)
        assert "params" in variables

        # 3 downsamples -> 32/8 = 4
        mean, logvar = model.apply(
            {"params": variables["params"]},
            x,
            method=model.encode,
        )
        assert mean.shape == (1, 4, 4, 4)
        assert logvar.shape == (1, 4, 4, 4)


class TestBuildingBlocks:
    """Tests for individual VAE components."""

    def test_resnet_block_same_channels(self):
        """ResNet block with no channel change."""
        block = vae.ResNetBlock(out_channels=64)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 8, 8, 64))
        params = block.init(key, x)["params"]
        out = block.apply({"params": params}, x)
        assert out.shape == (1, 8, 8, 64)
        assert "conv_shortcut" not in params

    def test_resnet_block_channel_change(self):
        """ResNet block with channel change creates shortcut."""
        block = vae.ResNetBlock(out_channels=128)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 8, 8, 64))
        params = block.init(key, x)["params"]
        out = block.apply({"params": params}, x)
        assert out.shape == (1, 8, 8, 128)
        assert "conv_shortcut" in params

    def test_downsample(self):
        """Downsample halves spatial dimensions."""
        ds = vae.Downsample(out_channels=64)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 16, 16, 64))
        params = ds.init(key, x)["params"]
        out = ds.apply({"params": params}, x)
        assert out.shape == (1, 8, 8, 64)

    def test_upsample(self):
        """Upsample doubles spatial dimensions."""
        us = vae.Upsample(out_channels=64)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 8, 8, 64))
        params = us.init(key, x)["params"]
        out = us.apply({"params": params}, x)
        assert out.shape == (1, 16, 16, 64)

    def test_attention_block(self):
        """AttentionBlock preserves shape."""
        attn = vae.AttentionBlock(channels=64)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 8, 8, 64))
        params = attn.init(key, x)["params"]
        out = attn.apply({"params": params}, x)
        assert out.shape == (1, 8, 8, 64)

    def test_mid_block(self):
        """MidBlock preserves shape."""
        mid = vae.MidBlock(channels=64)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 8, 8, 64))
        params = mid.init(key, x)["params"]
        out = mid.apply({"params": params}, x)
        assert out.shape == (1, 8, 8, 64)
