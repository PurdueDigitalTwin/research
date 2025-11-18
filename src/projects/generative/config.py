import functools

import fiddle as fdl
import optax

from learning.core import config as _config
from learning.data import cifar
from learning.data import preprocess
from learning.generative import meanflow


# MeanFlow Models
def meanflow_unet_cifar_10() -> _config.ExperimentConfig:
    return _config.ExperimentConfig(
        name="meanflow_unet_cifar_10",
        data=fdl.Partial(
            cifar.CIFAR10DataModule,
            preprocess_fn=preprocess.chain(
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
        model=fdl.Config(
            meanflow.MeanFlowUNetModel,
            in_channels=3,
            image_size=32,
            latent_channels=16,
            num_classes=10,
            use_cfg_embedding=False,
            dropout_rate=0.2,
            timestamp_cond="t_and_t_minus_r",
            timestamp_sampler="lognormal",
            timestamp_sampler_kwargs=dict(mean=-2.0, stddev=2.0),
            timestamp_overlap_rate=0.25,
            adaptive_weight_power=0.75,
        ),
        # TODO: implement the warmup in https://arxiv.org/abs/1706.02677
        batch_size=1024,
        lr_scheduler=fdl.Config(optax.constant_schedule, value=6e-4),
        optimizer=fdl.Partial(optax.adam, b1=0.9, b2=0.999),
        ema_rate=0.99995,
        num_train_steps=800_000,
    )
