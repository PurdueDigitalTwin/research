from datetime import datetime
import functools
import os
import platform
import typing

from absl import app
from absl import flags
from clu import metric_writers
from clu import platform as clu_platform
from fiddle import absl_flags
import fiddle as fdl
import jax
from jax import numpy as jnp
import jaxtyping
import optax
import tensorflow as tf
from tqdm import auto as tqdm
from tqdm.contrib import logging as tqdm_logging

from src.core import config as _config
from src.core import model as _model
from src.core import train as _train
from src.core import train_state as _train_state
from src.projects.generative.tools import fid
from src.utilities import logging
from src.utilities import visualization

CONFIG = absl_flags.DEFINE_fiddle_config(
    name="experiment",
    default=None,
    help_string="Experiment configuration.",
    required=True,
)
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="work_dir",
    default=None,
    help="Directory to store the experiment results.",
    required=True,
)
PyTree = jaxtyping.PyTree


# toggle off GPU/TPU for TensorFlow
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")
assert not tf.config.experimental.get_visible_devices("GPU")


def evaluate(
    rngs: jax.Array,
    model: _model.Model,
    params: PyTree,
    batch: typing.Dict[str, typing.Any],
    fid_metric: fid.FrechetInceptionDistance,
    **kwargs,
) -> _model.StepOutputs:
    r"""Conduct a single evaluation step and compute metrics."""

    def _generate(params: PyTree) -> jax.Array:
        local_rng = jax.random.fold_in(rngs, jax.lax.axis_index("batch"))
        outputs = model.forward(
            rngs=local_rng,
            params=params,
            deterministic=True,
            batch=batch,
            **kwargs,
        )
        assert isinstance(outputs, _model.StepOutputs)
        assert outputs.output is not None
        img = jnp.astype(
            jnp.clip(outputs.output * 0.5 + 0.5, 0.0, 1.0) * 255.0,
            jnp.uint8,
        )
        return img

    generate_fn = jax.pmap(_generate, axis_name="batch")
    with tqdm_logging.logging_redirect_tqdm():
        images, count = [], 0
        with tqdm.tqdm(total=50_000, unit="sample") as pbar:
            while count < 50_000:
                img = jnp.zeros_like(batch["image"])
                img = jnp.reshape(
                    img,
                    (jax.local_device_count(), -1) + img.shape[-3:],
                )
                out = generate_fn(params=params)
                out = jnp.reshape(out, (-1,) + out.shape[-3:])
                images.append(out)
                count += out.shape[0]
                pbar.update(out.shape[0])

    outputs = _model.StepOutputs()
    images = jnp.concatenate(images, axis=0)

    fid_score = fid_metric(images=images[0:50_000])
    outputs.scalars = {"fid": fid_score}

    img_grid = visualization.make_grid(
        images[0:32],
        n_rows=4,
        n_cols=8,
        padding=2,
    )
    outputs.images = {"sampled images": img_grid}

    return outputs


def training_step(
    rngs: jax.Array,
    model: _model.Model,
    state: _train_state.TrainState,
    batch: typing.Dict[str, typing.Any],
    **kwargs,
) -> typing.Tuple[_train_state.TrainState, _model.StepOutputs]:
    r"""Conduct a single training step and update train state."""
    local_rng = jax.random.fold_in(rngs, state.step)
    local_rng = jax.random.fold_in(local_rng, jax.lax.axis_index("batch"))

    def loss_fn(params: PyTree) -> typing.Tuple[jax.Array, _model.StepOutputs]:
        loss, outputs = model.compute_loss(
            rngs=local_rng,
            params=params,
            deterministic=False,
            batch=batch,
            **kwargs,
        )
        return loss, outputs

    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (_, outputs), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)

    return new_state, outputs


def main(_: typing.List[str]) -> int:
    r"""Main entry point for training and evaluate generative models."""
    del _  # unused.

    # Log the current platform
    logging.rank_zero_info("Running on platform: %s", platform.node())

    # Setup JAX runtime
    logging.rank_zero_info("Running on JAX backend: %s", jax.default_backend())
    logging.rank_zero_info(
        "Running on JAX process: %d / %d",
        jax.process_index() + 1,
        jax.process_count(),
    )
    logging.rank_zero_info("Running on JAX devices: %r", jax.devices())

    clu_platform.work_unit().set_task_status(
        "process_index: %d, process_count: %d"
        % (jax.process_index() + 1, jax.process_count()),
    )
    clu_platform.work_unit().create_artifact(
        clu_platform.ArtifactType.DIRECTORY,
        FLAGS.work_dir,
        "Working directory.",
    )

    # Setup Experiment
    exp_config = CONFIG.value
    if not isinstance(exp_config, _config.ExperimentConfig):
        logging.rank_zero_error(
            "Expect configuration to be of type `ExperimentConfig`, got %s.",
            type(exp_config),
        )
        return 1
    logging.rank_zero_info("Experiment Configuration:\n%s", exp_config)

    rng = jax.random.PRNGKey(exp_config.seed)
    log_dir = os.path.join(
        FLAGS.work_dir,
        exp_config.name,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    writer = metric_writers.create_default_writer(
        logdir=log_dir,
        just_logging=(jax.process_index() > 0),
    )

    logging.rank_zero_info("Building dataset...")
    rng, data_rng = jax.random.split(rng, num=2)
    p_datamodule = fdl.build(exp_config.data.module)
    datamodule = p_datamodule(
        batch_size=exp_config.data.batch_size,
        deterministic=exp_config.data.deterministic,
        drop_remainder=exp_config.data.drop_remainder,
        num_workers=exp_config.data.num_workers,
        rng=data_rng,
    )
    logging.rank_zero_info(
        "Building dataset %s... DONE!",
        datamodule.__class__.__name__,
    )

    logging.rank_zero_info("Building model...")
    rng, init_rng = jax.random.split(rng, num=2)
    p_model = fdl.build(exp_config.model)
    model = p_model(
        dtype=exp_config.dtype,
        param_dtype=exp_config.param_dtype,
        precision=exp_config.precision,
    )
    params = model.init(batch=None, rngs=init_rng)  # NOTE: use dummy batch
    logging.rank_zero_info(
        "Building model %s... DONE!",
        model.__class__.__name__,
    )

    logging.rank_zero_info("Building train state...")
    lr_scheduler = fdl.build(exp_config.optimizer.lr_schedule)
    p_optimizer = fdl.build(exp_config.optimizer.optimizer)
    tx = p_optimizer(learning_rate=lr_scheduler)
    if exp_config.optimizer.grad_clip_method == "norm":
        tx = optax.chain(
            optax.clip_by_global_norm(exp_config.optimizer.grad_clip_value),
            tx,
        )
    elif exp_config.optimizer.grad_clip_method == "value":
        tx = optax.chain(
            optax.clip(exp_config.optimizer.grad_clip_value),
            tx,
        )
    elif exp_config.optimizer.grad_clip_method is not None:
        logging.rank_zero_error(
            "Unknown grad clip method: %s",
            exp_config.optimizer.grad_clip_method,
        )
        return 1
    state = _train_state.TrainState.create(
        params=params,
        tx=tx,
        ema_rate=exp_config.optimizer.ema_rate,
    )
    logging.rank_zero_info("Building train state... DONE!")

    if exp_config.trainer.checkpoint_dir is not None:
        logging.rank_zero_error("Resuming from checkpoint not implemented.")
        return 1

    p_training_step = functools.partial(training_step, model=model)
    evaluation_fn = functools.partial(
        evaluate,
        model=model,
        rngs=rng,
        batch=next(datamodule.eval_dataloader()),
        # TODO (juanwu): make `fid_metric` configurable
        fid_metric=fid.FrechetInceptionDistance(
            train_dataset=datamodule._hf_dataset["train"],  # type: ignore
            image_key="img",
        ),
    )
    if exp_config.mode == "train":
        _train.run(
            state=state,
            datamodule=datamodule,
            training_step=p_training_step,
            evaluation_fn=evaluation_fn,
            num_train_steps=exp_config.trainer.num_train_steps,
            writer=writer,
            work_dir=log_dir,
            rng=rng,
            checkpoint_every_n_steps=exp_config.trainer.checkpoint_every_n_steps,
            log_every_n_steps=exp_config.trainer.log_every_n_steps,
            eval_every_n_steps=exp_config.trainer.eval_every_n_steps,
            profile=exp_config.trainer.profile,
        )
    elif exp_config.mode == "evaluate":
        evaluation_fn(params=state.ema_params)
    else:
        logging.rank_zero_error("Mode %s not implemented.", exp_config.mode)
        return 1

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
