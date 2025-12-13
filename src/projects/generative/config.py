import functools
import math

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
            batch_size=1024,
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
            timestamp_cond="t_and_t_minus_r",
            timestamp_sampler="logit-normal",
            timestamp_sampler_kwargs=dict(mean=-2.0, stddev=2.0),
            timestamp_overlap_rate=0.25,
            adaptive_weight_power=0.75,
        ),
        trainer=_config.TrainerConfig(
            num_train_steps=800_000,
            log_every_n_steps=50,
            checkpoint_every_n_steps=10_000,  # save every 10k steps
            eval_every_n_steps=1_000,
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
            ema_rate=0.9999,
        ),
        seed=42,
    )
