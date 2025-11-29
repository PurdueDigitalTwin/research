import functools

import fiddle as fdl
import optax

from src.core import config as _config
from src.data import huggingface
from src.data import preprocess
from src.projects.generative import meanflow


# ==============================================================================
# MeanFlow Models
def meanflow_unet_cifar_10() -> _config.ExperimentConfig:
    return _config.ExperimentConfig(
        name="meanflow_unet_cifar_10",
        mode="train",
        data=_config.DataConfig(
            module=fdl.Partial(
                huggingface.CIFAR10DataModule,
                resize=32,
                transform=preprocess.chain(
                    functools.partial(
                        preprocess.filter_keys,
                        keys=["image", "label"],
                    ),
                    functools.partial(
                        preprocess.normalize,
                        mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5),
                    ),
                ),
            ),
            batch_size=128,
            num_workers=4,
            deterministic=True,
            drop_remainder=True,
        ),
        model=fdl.Partial(
            meanflow.MeanFlowUNetModel,
            in_channels=3,
            image_size=32,
            latent_channels=128,
            num_classes=10,
            use_cfg_embedding=False,
            dropout_rate=0.2,
            timestamp_cond="t_and_t_minus_r",
            timestamp_sampler="lognormal",
            timestamp_sampler_kwargs=dict(mean=-2.0, stddev=2.0),
            timestamp_overlap_rate=0.25,
            adaptive_weight_power=0.75,
        ),
        trainer=_config.TrainerConfig(
            num_train_steps=800_000,
            log_every_n_steps=5,
            checkpoint_every_n_steps=10_000,  # save every 10k steps
            eval_every_n_steps=1_000_000,  # NOTE: never evaluate now
            max_checkpoints_to_keep=3,
            profile=False,
        ),
        optimizer=_config.OptimizerConfig(
            lr_schedule=fdl.Config(optax.constant_schedule, value=6e-4),
            optimizer=fdl.Partial(optax.adam, b1=0.9, b2=0.999),
            ema_rate=0.9999,
        ),
        seed=42,
    )
