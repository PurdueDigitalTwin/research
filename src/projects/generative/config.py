import functools
import math
import os

import datasets
import fiddle as fdl
import jax
import optax

from src.core import config as _config
from src.data import huggingface
from src.data import preprocess
from src.projects.generative import ddpm
from src.projects.generative import meanflow
from src.projects.generative.tools import fid


# ==============================================================================
# Denoising Deep Probabilistic Models (DDPM)
def ddpm_unet_cifar_10() -> _config.ExperimentConfig:
    return _config.ExperimentConfig(
        project_name="ddpm",
        exp_name="unet_cifar_10",
        mode="train",
        data=_config.DataConfig(
            module=fdl.Partial(
                huggingface.CIFAR10DataModule,
                shuffle_buffer_size=50_000,
                transform=preprocess.chain(
                    functools.partial(
                        preprocess.filter_keys,
                        keys=["image", "label"],
                    ),
                    functools.partial(
                        preprocess.normalize,
                        mean=(0.0, 0.0, 0.0),
                        std=(1.0, 1.0, 1.0),
                    ),
                ),
                use_cache=True,
            ),
            batch_size=128,
            num_workers=4,
            deterministic=True,
            drop_remainder=True,
        ),
        model=fdl.Partial(
            ddpm.DDPMGaussianUNetModel,
            in_channels=3,
            image_size=32,
            features=128,
            ch_mults=[1, 2, 2, 2],
            dropout_rate=0.1,
            epsilon=1e-6,
            attn_resolutions=[16],
            num_res_blocks=2,
            resample_with_conv=True,
            predict_variance=False,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            model_var_type="fixed_large",
            num_diffusion_steps=1_000,
        ),
        metric=fdl.Config(
            fid.FrechetInceptionDistance,
            dataset=functools.partial(
                datasets.load_dataset,
                path="uoft-cs/cifar10",
                token=os.getenv("HF_TOKEN", None),
                revision="0b2714987fa478483af9968de7c934580d0bb9a2",
                split="train",
            ),
            image_key="img",
            batch_size=32,
        ),
        trainer=_config.TrainerConfig(
            num_train_steps=800_000,
            log_every_n_steps=50,
            checkpoint_every_n_steps=10_000,  # save every 10k steps
            eval_every_n_steps=50_000,  # slow evaluate every 50k steps
            max_checkpoints_to_keep=3,
            profile=False,
        ),
        optimizer=_config.OptimizerConfig(
            lr_schedule=fdl.Config(
                optax.warmup_constant_schedule,
                init_value=1e-8,
                peak_value=2e-4,
                warmup_steps=5_000,
            ),
            optimizer=fdl.Partial(optax.adam, b1=0.9, b2=0.999),
            grad_clip_method="norm",
            grad_clip_value=1.0,
            ema_rate=0.9999,
        ),
        seed=42,
        dtype=jax.numpy.float32,
        param_dtype=jax.numpy.float32,
        precision=jax.lax.Precision.HIGHEST,
    )


# ==============================================================================
# MeanFlow Models
def meanflow_unet_cifar_10() -> _config.ExperimentConfig:
    return _config.ExperimentConfig(
        project_name="meanflow",
        exp_name="unet_cifar_10",
        mode="train",
        data=_config.DataConfig(
            module=fdl.Partial(
                huggingface.CIFAR10DataModule,
                shuffle_buffer_size=50_000,
                transform=preprocess.chain(
                    functools.partial(
                        preprocess.filter_keys,
                        keys=["image", "label"],
                    ),
                    functools.partial(
                        preprocess.normalize,
                        mean=(0.0, 0.0, 0.0),
                        std=(1.0, 1.0, 1.0),
                    ),
                ),
                use_cache=True,
            ),
            batch_size=64,
            num_workers=2,
            deterministic=True,
            drop_remainder=True,
        ),
        model=fdl.Partial(
            meanflow.MeanFlowUNetModel,
            in_channels=3,
            image_size=32,
            features=128,
            dropout_rate=0.2,
            epsilon=1e-6,
            skip_scale=math.sqrt(0.5),
            resample_filter=[1, 3, 3, 1],
            timestamp_cond="t_and_t_minus_r",
            timestamp_sampler="logit-normal",
            timestamp_sampler_kwargs=dict(
                mean=-0.6,
                stddev=1.6,
                r_mean=-4.0,
                r_stddev=1.6,
            ),
            timestamp_overlap_rate=0.25,
            timestamp_sampler_version="v1",
            adaptive_weight_power=0.75,
        ),
        metric=fdl.Config(
            fid.FrechetInceptionDistance,
            dataset=functools.partial(
                datasets.load_dataset,
                path="uoft-cs/cifar10",
                token=os.getenv("HF_TOKEN", None),
                revision="0b2714987fa478483af9968de7c934580d0bb9a2",
                split="train",
            ),
            image_key="img",
            batch_size=32,
        ),
        trainer=_config.TrainerConfig(
            num_train_steps=800_000,
            log_every_n_steps=50,
            checkpoint_every_n_steps=10_000,  # save every 10k steps
            eval_every_n_steps=2_500,
            max_checkpoints_to_keep=3,
            profile=False,
        ),
        optimizer=_config.OptimizerConfig(
            lr_schedule=fdl.Config(
                optax.warmup_constant_schedule,
                init_value=1e-8,
                peak_value=6e-4,
                warmup_steps=10_000,
            ),
            optimizer=fdl.Partial(optax.adam, b1=0.9, b2=0.999),
            ema_rate=0.99995,
            ema_update_period=16,
        ),
        seed=42,
        dtype=jax.numpy.float32,
        param_dtype=jax.numpy.float32,
        precision=jax.lax.Precision.HIGHEST,
    )


def meanflow_dit_imagenet_256() -> _config.ExperimentConfig:
    r"""MeanFlow with DiT-B/2 on ImageNet 256x256 (latent space).

    Following the official MeanFlow paper (arXiv:2505.13447): DiT-B/2 (768 hidden, 12 depth, 12
    heads), latent-space training with SD VAE (pcuenq/sd-vae-ft-mse-flax).
    """
    return _config.ExperimentConfig(
        project_name="meanflow",
        exp_name="dit_imagenet_256",
        mode="train",
        data=_config.DataConfig(
            module=fdl.Partial(
                huggingface.ImageNet1KDataModule,
                shuffle_buffer_size=10_000,
                transform=preprocess.chain(
                    functools.partial(
                        preprocess.filter_keys,
                        keys=["image", "label"],
                    ),
                    functools.partial(
                        preprocess.resize,
                        size=(256, 256),
                    ),
                    functools.partial(
                        preprocess.normalize,
                        mean=(0.0, 0.0, 0.0),
                        std=(1.0, 1.0, 1.0),
                    ),
                ),
                streaming=True,
                data_dir=os.getenv(
                    "IMAGENET_DATA_DIR",
                    "gs://pdt_training/juanwu/cache/huggingface"
                    "/imagenet-1k",
                ),
            ),
            batch_size=256,
            num_workers=4,
            deterministic=True,
            drop_remainder=True,
        ),
        model=fdl.Partial(
            meanflow.MeanFlowDiTModel,
            in_channels=4,
            image_size=32,
            features=768,
            patch_size=2,
            depth=12,
            num_heads=12,
            ffn_ratio=4,
            dropout_rate=0.1,
            num_classes=1000,
            class_dropout_prob=0.1,
            cfg_omega=1.0,
            cfg_kappa=0.5,
            epsilon=1e-6,
            timestamp_cond="t_and_t_minus_r",
            timestamp_sampler="logit-normal",
            timestamp_sampler_kwargs=dict(mean=-0.4, stddev=1.0),
            timestamp_overlap_rate=0.75,
            adaptive_weight_power=1.0,
            vae_path=os.getenv("VAE_PATH", "pcuenq/sd-vae-ft-mse-flax"),
            vae_scaling_factor=0.18215,
        ),
        metric=fdl.Config(
            fid.FrechetInceptionDistance,
            dataset=functools.partial(
                datasets.load_dataset,
                path="ILSVRC/imagenet-1k",
                token=os.getenv("HF_TOKEN", None),
                revision=("49e2ee26f3810fb5a7536bbf" "732a7b07389a47b5"),
                split="train",
                streaming=True,
            ),
            image_key="image",
            batch_size=32,
            ref_cache_path=os.getenv(
                "FID_REF_CACHE",
                "gs://pdt_training/juanwu/cache"
                "/imagenet-1k-fid-ref-stats.npz",
            ),
        ),
        trainer=_config.TrainerConfig(
            num_train_steps=800_000,
            log_every_n_steps=50,
            checkpoint_every_n_steps=10_000,
            eval_every_n_steps=5_000,
            max_checkpoints_to_keep=3,
            profile=False,
        ),
        optimizer=_config.OptimizerConfig(
            lr_schedule=fdl.Config(
                optax.warmup_constant_schedule,
                init_value=1e-8,
                peak_value=1e-4,
                warmup_steps=10_000,
            ),
            optimizer=fdl.Partial(optax.adam, b1=0.9, b2=0.95),
            ema_rate=0.9999,
        ),
        seed=42,
        dtype=jax.numpy.float32,
        param_dtype=jax.numpy.float32,
        precision=jax.lax.Precision.HIGHEST,
    )


def meanflow_dit_imagenet_256_latent() -> _config.ExperimentConfig:
    r"""MeanFlow DiT-B/4 on ImageNet 256x256 with pre-encoded latents.

    Aligned with official MeanFlow implementation
    (github.com/Gsunshine/meanflow, ``run_b4.yml``).
    Uses DiT-B/4 (patch_size=4) for ablation experiments
    (Table 4 of arXiv:2505.13447).

    Key settings matching official:
      - patch_size=4, no dropout, no LR warmup
      - norm_eps=0.01, adamw with wd=0, constant LR=1e-4
      - EMA 0.9999, logit-normal t/r, overlap_rate=0.75

    Steps = 240 epochs times `floor(1_281_167 / 1024)` is approximately 300K.
    """
    return _config.ExperimentConfig(
        project_name="meanflow",
        exp_name="dit_b4_imagenet_256_latent",
        mode="train",
        data=_config.DataConfig(
            module=fdl.Partial(
                huggingface.ImageNetLatentDataModule,
                shuffle_buffer_size=10_000,
                data_dir=os.getenv(
                    "IMAGENET_LATENT_DIR",
                    "gs://pdt_training/juanwu/cache" "/imagenet-1k-latent",
                ),
            ),
            batch_size=256,
            num_workers=4,
            deterministic=True,
            drop_remainder=True,
        ),
        model=fdl.Partial(
            meanflow.MeanFlowDiTModel,
            in_channels=4,
            image_size=32,
            features=768,
            patch_size=4,
            depth=12,
            num_heads=12,
            ffn_ratio=4,
            dropout_rate=0.0,
            num_classes=1000,
            class_dropout_prob=0.1,
            cfg_omega=1.0,
            cfg_kappa=0.5,
            epsilon=1e-6,
            norm_eps=0.01,
            timestamp_cond="t_and_t_minus_r",
            timestamp_sampler="logit-normal",
            timestamp_sampler_kwargs=dict(mean=-0.4, stddev=1.0),
            timestamp_overlap_rate=0.75,
            adaptive_weight_power=1.0,
            vae_path=os.getenv("VAE_PATH", "pcuenq/sd-vae-ft-mse-flax"),
            vae_scaling_factor=0.18215,
        ),
        metric=fdl.Config(
            fid.FrechetInceptionDistance,
            dataset=functools.partial(
                datasets.load_dataset,
                path="ILSVRC/imagenet-1k",
                token=os.getenv("HF_TOKEN", None),
                revision=("49e2ee26f3810fb5a7536bbf" "732a7b07389a47b5"),
                split="train",
                streaming=True,
            ),
            image_key="image",
            batch_size=32,
            ref_cache_path=os.getenv(
                "FID_REF_CACHE",
                "gs://pdt_training/juanwu/cache"
                "/imagenet-1k-fid-ref-stats.npz",
            ),
        ),
        trainer=_config.TrainerConfig(
            num_train_steps=300_000,
            log_every_n_steps=50,
            checkpoint_every_n_steps=10_000,
            eval_every_n_steps=5_000,
            max_checkpoints_to_keep=3,
            profile=False,
        ),
        optimizer=_config.OptimizerConfig(
            lr_schedule=fdl.Config(
                optax.warmup_constant_schedule,
                init_value=1e-4,
                peak_value=1e-4,
                warmup_steps=0,
            ),
            optimizer=fdl.Partial(
                optax.adamw,
                b1=0.9,
                b2=0.95,
                weight_decay=0,
            ),
            ema_rate=0.9999,
        ),
        seed=42,
        dtype=jax.numpy.float32,
        param_dtype=jax.numpy.float32,
        precision=jax.lax.Precision.HIGHEST,
    )


def vamf_tw_dit_imagenet_256_latent() -> _config.ExperimentConfig:
    r"""VaMF (trace-weighted) DiT-B/4 on ImageNet 256x256 latents.

    Identical to ``meanflow_dit_imagenet_256_latent`` except the
    per-sample loss is multiplied by the variance-aware trace weight
    ``1 / (1 + sigma_t * tr(BB^T) / d)`` with ``B = (t-r) J - I`` and
    the default ``sigma_t = t^2`` schedule. Adaptive loss weighting is
    disabled to avoid double-weighting.
    """
    config = meanflow_dit_imagenet_256_latent()
    config.exp_name = "vamf_tw_dit_b4_imagenet_256_latent"
    # Replace MeanFlow model with VaMF-TW variant.
    config.model.adaptive_weight_power = 0.0
    config.model.use_trace_weight = True
    config.model.tw_n_probes = 1
    config.model.tw_sigma_schedule = "t_squared"
    return config


def vamf_l2_dit_imagenet_256_latent() -> _config.ExperimentConfig:
    r"""VaMF-L2 DiT-B/4 on ImageNet 256x256 latents.

    Realizes the deterministic-tangent corner (β = 1) of the
    control-variate trade-off (Theorem 3 of the paper): replaces
    the JVP tangent ``v_g`` (CFG-mixed conditional velocity) with
    its EMA-derived counterpart, eliminating the Jacobian-amplified
    gradient variance ``g^T J Σ J^T g``. Adds an FM anchor at small
    ``(t - r)`` to control the residual EMA-tracking bias from
    Theorem 2. No trace weight (the EMA tangent removes the
    dominant variance term that the trace weight reweights).

    Identical to ``meanflow_dit_imagenet_256_latent`` except:
      - ``ema_tangent=True``
      - ``fm_anchor_weight=0.1`` (boundary supervision)
      - ``adaptive_weight_power=0.0`` (avoid double weighting)
      - ``use_trace_weight=False``
    """
    config = meanflow_dit_imagenet_256_latent()
    config.exp_name = "vamf_l2_dit_b4_imagenet_256_latent"
    config.model.adaptive_weight_power = 0.0
    config.model.use_trace_weight = False
    config.model.ema_tangent = True
    config.model.fm_anchor_weight = 0.1
    config.model.fm_anchor_delta_min = 0.0
    config.model.fm_anchor_delta_max = 1e-3
    return config


def vamf_beta05_dit_imagenet_256_latent() -> _config.ExperimentConfig:
    r"""DiT-B/4 ImageNet-256 latent at the interior tangent-mixing point β=0.5.

    Identical to ``meanflow_dit_imagenet_256_latent`` (the baseline)
    except ``tangent_beta=0.5`` — the JVP tangent is the average of
    the vanilla MeanFlow CFG-mixed velocity (computed under
    stop-gradient of the current params) and the VaMF-L2 EMA-derived
    velocity. Together with the baseline (β=0) and VaMF-L2 (β=1)
    runs, this provides the third datapoint needed to validate
    Theorem 3's interior-optimum prediction at DiT scale.

    All other hyperparameters (adaptive_weight_power=1.0, no FM
    anchor, etc.) match the baseline so this run isolates the effect
    of β alone.
    """
    config = meanflow_dit_imagenet_256_latent()
    config.exp_name = "vamf_beta05_dit_b4_imagenet_256_latent"
    config.model.tangent_beta = 0.5
    return config


def vamf_l2_aw1_dit_imagenet_256_latent() -> _config.ExperimentConfig:
    r"""VaMF-L2 DiT-B/4 on ImageNet 256x256, with Karras adaptive weighting.

    Identical to ``vamf_l2_dit_imagenet_256_latent`` except
    ``adaptive_weight_power=1.0`` (matching the baseline). Isolates the
    effect of the deterministic-tangent / EMA-tangent / FM-anchor stack
    from the effect of disabling adaptive loss weighting. Pairs with
    the standard VaMF-L2 config to form a 3-way ablation:

      - ``meanflow_dit_imagenet_256_latent``: baseline (stochastic
        tangent, adaptive_weight=1.0).
      - ``vamf_l2_dit_imagenet_256_latent``: VaMF-L2 (deterministic
        tangent, adaptive_weight=0.0).
      - ``vamf_l2_aw1_dit_imagenet_256_latent``: VaMF-L2 + matched
        adaptive weighting (deterministic tangent, adaptive_weight=1.0).

    If the FID gap (baseline vs VaMF-L2) shrinks in this config, the
    gap was driven by the missing adaptive weighting; if it persists,
    the gap is genuinely from the deterministic tangent (β=1 corner).
    """
    config = vamf_l2_dit_imagenet_256_latent()
    config.exp_name = "vamf_l2_aw1_dit_b4_imagenet_256_latent"
    config.model.adaptive_weight_power = 1.0
    return config


def meanflow_dit_afhqv2_256_pixel() -> _config.ExperimentConfig:
    r"""MeanFlow DiT-B/4 on AFHQv2 256x256 pixel space (online VAE).

    Smaller-scale converged-FID companion to the ImageNet-256 latent
    runs. AFHQv2 has ~15k training images across 3 classes (cat, dog,
    wild) — small enough that we can encode online via the SD VAE
    inside ``MeanFlowDiTModel.training_step`` without precomputing
    latents. Same DiT-B/4 backbone, same hyperparameters, just a
    different dataset and ``num_classes=3`` (plus null token).

    Total samples per epoch: ~15k / batch_size=64 ≈ 234 steps. We
    target 200 epochs ≈ 47k steps, with FID checkpoint cadence
    matching the ImageNet config scaled down 6x.
    """
    config = meanflow_dit_imagenet_256_latent()
    config.exp_name = "dit_b4_afhqv2_256_pixel"
    # Swap data: pixel-space AFHQv2 (online VAE in training_step).
    config.data = _config.DataConfig(
        module=fdl.Partial(
            huggingface.AFHQv2DataModule,
            image_size=256,
            shuffle_buffer_size=10_000,
        ),
        batch_size=64,
        num_workers=4,
        deterministic=True,
        drop_remainder=True,
    )
    # AFHQv2 has 3 classes (cat / dog / wild).
    config.model.num_classes = 3
    # Reduced training horizon for the smaller dataset.
    config.trainer.num_train_steps = 50_000
    config.trainer.checkpoint_every_n_steps = 2_500
    config.trainer.eval_every_n_steps = 2_500
    # FID over AFHQv2 train split (no canonical eval set here; we use
    # a freshly precomputed reference cache).
    config.metric = fdl.Config(
        fid.FrechetInceptionDistance,
        dataset=functools.partial(
            datasets.load_dataset,
            path="huggan/AFHQv2",
            token=os.getenv("HF_TOKEN", None),
            revision="f638548a7eccf134045249ed2ac708505bac6e2e",
            split="train",
            streaming=True,
        ),
        image_key="image",
        batch_size=32,
        ref_cache_path=os.getenv(
            "FID_REF_CACHE_AFHQV2",
            "gs://pdt_training/juanwu/cache/afhqv2-fid-ref-stats.npz",
        ),
    )
    return config


def vamf_l2_dit_afhqv2_256_pixel() -> _config.ExperimentConfig:
    r"""VaMF-L2 DiT-B/4 on AFHQv2 256x256 pixel space.

    Pairs with ``meanflow_dit_afhqv2_256_pixel`` for the AFHQ
    converged-FID comparison (Option H in the overnight plan).
    Same VaMF-L2 deltas as ``vamf_l2_dit_imagenet_256_latent``:
    EMA tangent + FM anchor, no trace weight, no adaptive
    weighting.
    """
    config = meanflow_dit_afhqv2_256_pixel()
    config.exp_name = "vamf_l2_dit_b4_afhqv2_256_pixel"
    config.model.adaptive_weight_power = 0.0
    config.model.use_trace_weight = False
    config.model.ema_tangent = True
    config.model.fm_anchor_weight = 0.1
    config.model.fm_anchor_delta_min = 0.0
    config.model.fm_anchor_delta_max = 1e-3
    return config


def improved_meanflow_unet_cifar_10() -> _config.ExperimentConfig:
    r"""Improved MeanFlow (iMF) with U-Net on CIFAR-10."""
    return _config.ExperimentConfig(
        project_name="meanflow",
        exp_name="iMF_unet_cifar_10",
        mode="train",
        data=_config.DataConfig(
            module=fdl.Partial(
                huggingface.CIFAR10DataModule,
                shuffle_buffer_size=50_000,
                transform=preprocess.chain(
                    functools.partial(
                        preprocess.filter_keys,
                        keys=["image", "label"],
                    ),
                    functools.partial(
                        preprocess.normalize,
                        mean=(0.0, 0.0, 0.0),
                        std=(1.0, 1.0, 1.0),
                    ),
                ),
                use_cache=True,
            ),
            batch_size=128,
            num_workers=2,
            deterministic=True,
            drop_remainder=True,
        ),
        model=fdl.Partial(
            meanflow.ImprovedMeanFlowUNetModel,
            in_channels=3,
            image_size=32,
            features=128,
            dropout_rate=0.2,
            epsilon=1e-6,
            skip_scale=math.sqrt(0.5),
            resample_filter=[1, 3, 3, 1],
            timestamp_cond="t_and_t_minus_r",
            timestamp_sampler="logit-normal",
            timestamp_sampler_kwargs=dict(mean=-0.4, stddev=1.0),
            timestamp_overlap_rate=0.5,
            adaptive_weight_power=0.75,
        ),
        metric=fdl.Config(
            fid.FrechetInceptionDistance,
            dataset=functools.partial(
                datasets.load_dataset,
                path="uoft-cs/cifar10",
                token=os.getenv("HF_TOKEN", None),
                revision="0b2714987fa478483af9968de7c934580d0bb9a2",
                split="train",
            ),
            image_key="img",
            batch_size=32,
        ),
        trainer=_config.TrainerConfig(
            num_train_steps=800_000,
            log_every_n_steps=50,
            checkpoint_every_n_steps=10_000,  # save every 10k steps
            eval_every_n_steps=2_500,
            max_checkpoints_to_keep=3,
            profile=False,
        ),
        optimizer=_config.OptimizerConfig(
            lr_schedule=fdl.Config(
                optax.warmup_constant_schedule,
                init_value=1e-8,
                peak_value=1e-4,
                warmup_steps=10_000,
            ),
            optimizer=fdl.Partial(optax.adam, b1=0.9, b2=0.999),
            ema_rate=0.99995,
        ),
        seed=42,
        dtype=jax.numpy.float32,
        param_dtype=jax.numpy.float32,
        precision=jax.lax.Precision.HIGHEST,
    )


# ==============================================================================
# Variance-Aware MeanFlow (VaMF)
def vamf_unet_cifar_10() -> _config.ExperimentConfig:
    r"""VaMF with EMA tangent and FM anchor."""
    return _config.ExperimentConfig(
        project_name="meanflow",
        exp_name="vamf_unet_cifar_10",
        mode="train",
        data=_config.DataConfig(
            module=fdl.Partial(
                huggingface.CIFAR10DataModule,
                shuffle_buffer_size=50_000,
                transform=preprocess.chain(
                    functools.partial(
                        preprocess.filter_keys,
                        keys=["image", "label"],
                    ),
                    functools.partial(
                        preprocess.normalize,
                        mean=(0.0, 0.0, 0.0),
                        std=(1.0, 1.0, 1.0),
                    ),
                ),
                use_cache=True,
            ),
            batch_size=128,
            num_workers=2,
            deterministic=True,
            drop_remainder=True,
        ),
        model=fdl.Partial(
            meanflow.VAMeanFlowUNetModel,
            in_channels=3,
            image_size=32,
            features=128,
            dropout_rate=0.2,
            epsilon=1e-6,
            skip_scale=math.sqrt(0.5),
            resample_filter=[1, 3, 3, 1],
            timestamp_cond="t_and_t_minus_r",
            timestamp_sampler="logit-normal",
            timestamp_sampler_kwargs=dict(mean=-0.4, stddev=1.0),
            timestamp_overlap_rate=0.5,
            adaptive_weight_power=0.0,
            snr_epsilon=1e-2,
            fm_anchor_weight=0.5,
            fm_anchor_delta_min=1e-4,
            fm_anchor_delta_max=0.01,
            predict_variance=False,
            no_fm_anchor=False,
            boundary_tangent=False,
        ),
        metric=fdl.Config(
            fid.FrechetInceptionDistance,
            dataset=functools.partial(
                datasets.load_dataset,
                path="uoft-cs/cifar10",
                token=os.getenv("HF_TOKEN", None),
                revision="0b2714987fa478483af9968de7c934580d0bb9a2",
                split="train",
            ),
            image_key="img",
            batch_size=32,
        ),
        trainer=_config.TrainerConfig(
            num_train_steps=150_000,
            log_every_n_steps=50,
            checkpoint_every_n_steps=10_000,
            eval_every_n_steps=2_500,
            max_checkpoints_to_keep=3,
            profile=False,
        ),
        optimizer=_config.OptimizerConfig(
            lr_schedule=fdl.Config(
                optax.warmup_constant_schedule,
                init_value=1e-8,
                peak_value=1e-4,
                warmup_steps=10_000,
            ),
            optimizer=fdl.Partial(optax.adam, b1=0.9, b2=0.999),
            ema_rate=0.99995,
        ),
        seed=42,
        dtype=jax.numpy.float32,
        param_dtype=jax.numpy.float32,
        precision=jax.lax.Precision.HIGHEST,
    )


def vamf_nll_unet_cifar_10() -> _config.ExperimentConfig:
    r"""VaMF with heteroscedastic variance prediction."""
    return _config.ExperimentConfig(
        project_name="meanflow",
        exp_name="vamf_nll_unet_cifar_10",
        mode="train",
        data=_config.DataConfig(
            module=fdl.Partial(
                huggingface.CIFAR10DataModule,
                shuffle_buffer_size=50_000,
                transform=preprocess.chain(
                    functools.partial(
                        preprocess.filter_keys,
                        keys=["image", "label"],
                    ),
                    functools.partial(
                        preprocess.normalize,
                        mean=(0.0, 0.0, 0.0),
                        std=(1.0, 1.0, 1.0),
                    ),
                ),
                use_cache=True,
            ),
            batch_size=128,
            num_workers=2,
            deterministic=True,
            drop_remainder=True,
        ),
        model=fdl.Partial(
            meanflow.VAMeanFlowUNetModel,
            in_channels=3,
            image_size=32,
            features=128,
            dropout_rate=0.2,
            epsilon=1e-6,
            skip_scale=math.sqrt(0.5),
            resample_filter=[1, 3, 3, 1],
            timestamp_cond="t_and_t_minus_r",
            timestamp_sampler="logit-normal",
            timestamp_sampler_kwargs=dict(mean=-0.4, stddev=1.0),
            timestamp_overlap_rate=0.5,
            adaptive_weight_power=0.0,
            fm_anchor_weight=0.5,
            fm_anchor_delta_min=1e-4,
            fm_anchor_delta_max=0.01,
            predict_variance=True,
            variance_floor=1e-4,
            nll_warmup_steps=10_000,
            nll_ramp_steps=2_000,
            no_fm_anchor=False,
            boundary_tangent=False,
        ),
        metric=fdl.Config(
            fid.FrechetInceptionDistance,
            dataset=functools.partial(
                datasets.load_dataset,
                path="uoft-cs/cifar10",
                token=os.getenv("HF_TOKEN", None),
                revision="0b2714987fa478483af9968de7c934580d0bb9a2",
                split="train",
            ),
            image_key="img",
            batch_size=32,
        ),
        trainer=_config.TrainerConfig(
            num_train_steps=150_000,
            log_every_n_steps=50,
            checkpoint_every_n_steps=10_000,
            eval_every_n_steps=2_500,
            max_checkpoints_to_keep=3,
            profile=False,
        ),
        optimizer=_config.OptimizerConfig(
            lr_schedule=fdl.Config(
                optax.warmup_constant_schedule,
                init_value=1e-8,
                peak_value=1e-4,
                warmup_steps=10_000,
            ),
            optimizer=fdl.Partial(optax.adam, b1=0.9, b2=0.999),
            ema_rate=0.99995,
        ),
        seed=42,
        dtype=jax.numpy.float32,
        param_dtype=jax.numpy.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
