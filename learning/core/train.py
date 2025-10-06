import collections
import copy
import dataclasses
from datetime import datetime
import functools
import os
import typing

from clu import checkpoint
from clu import metric_writers
from clu import periodic_actions
import fiddle as fdl
from flax import jax_utils
from flax import struct
from flax.core import frozen_dict
import jax
import jaxtyping
import optax

from learning.core import config as _config
from learning.core import mixin as _mixin
from learning.utilities import logging

# Constants
PyTree = jaxtyping.PyTree


# Helper functions
def shard(tree: PyTree) -> PyTree:
    """Helper function for `jax.pmap` to shard a pytree onto local devices.

    Args:
        tree (PyTree): A pytree (e.g., nested dict/list/tuple of arrays) containing data to be sharded across local devices.
    """
    _shape_prefix = (jax.local_device_count(), -1)
    return jax.tree_util.tree_map(
        lambda x: (
            x.reshape(_shape_prefix + x.shape[1:])
            if hasattr(x, "reshape")
            else x
        ),
        tree=tree,
    )


class TrainState(struct.PyTreeNode):
    """Train state with exponential moving average of params."""

    step: int
    """int: Counter incremented by calls to `apply_gradients()`."""
    params: frozen_dict.FrozenDict = struct.field(pytree_node=True)
    """FrozenDict: The model parameters to be updated by optimizer."""
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    """optax.GradientTransformation: The Optax optimizer."""
    opt_state: optax.OptState = struct.field(pytree_node=True)
    """OptState: The state of the Optax optimizer."""
    model: _mixin.ModelMixin = struct.field(pytree_node=False)
    """ModelMixin: The model being trained."""
    ema_params: frozen_dict.FrozenDict = struct.field(pytree_node=True)
    """FrozenDict: Exponential moving average of params."""
    ema_rate: float = 0.999
    """float: Decay rate for exponential moving average of params."""

    def apply_gradients(self, *, grads: PyTree, **kwargs) -> "TrainState":
        """Applies a single gradient step and update parameters.

        Args:
            grads (PyTree): Gradients with the same structure as `.params`.
            kwargs: Additional dataclass attributes to be `.replace()`-ed.

        Returns:
            TrainState: A new state with updated parameters and optimizer state.
        """
        updates, new_opt_state = self.tx.update(
            updates=grads,
            state=self.opt_state,
            params=self.params,
        )
        new_params = optax.apply_updates(params=self.params, updates=updates)
        new_ema_params = jax.tree_map(
            lambda x, y: x + (1.0 - self.ema_rate) * (y - x),
            self.ema_params,
            new_params,
        )

        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(
        cls: typing.Type["TrainState"],
        *,
        model: _mixin.ModelMixin,
        params: PyTree,
        tx: optax.GradientTransformation,
        ema_rate: float = 0.999,
        **kwargs,
    ) -> "TrainState":
        """Creates a new instance of `TrainState`."""

        opt_state = tx.init(params=params)
        return cls(
            step=0,
            model=model,
            params=params,
            tx=tx,
            opt_state=opt_state,
            ema_params=copy.deepcopy(params),
            ema_rate=ema_rate,
            **kwargs,
        )


def training_step(
    base_rng: jax.random.KeyArray,
    model: _mixin.ModelMixin,
    state: TrainState,
    batch: typing.Dict[str, typing.Any],
    **kwargs,
) -> typing.Tuple[TrainState, PyTree]:
    """Conducts a single training step and update train state."""
    rng = jax.random.fold_in(
        key=base_rng,
        data=jax.lax.axis_index("batch"),  # type: ignore
    )
    rng = jax.random.fold_in(key=rng, data=state.step)

    def loss_fn(params: PyTree) -> typing.Tuple[jax.Array, PyTree]:
        outputs = model.compute_loss(
            rngs=rng,
            params=params,
            deterministic=False,
            **batch,
            **kwargs,
        )
        outputs = dataclasses.asdict(outputs)
        assert "loss" in outputs, "Model must return a loss."
        loss = outputs.pop("loss")

        return loss, outputs

    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, outputs), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)

    scalar_outputs = {
        key: value
        for key, value in outputs.items()
        if (
            (not isinstance(value, typing.Iterable))
            or (isinstance(value, jax.Array) and value.ndim == 0)
        )
    }
    metrics = {"loss": loss, **scalar_outputs}
    metrics = jax.tree_util.tree_map(
        lambda x: jax.lax.pmean(x, axis_name="batch"), metrics
    )

    return new_state, metrics


def train_and_evaluate(
    config: _config.ExperimentConfig,
    work_dir: str,
) -> None:
    """Runs training and evaluation loop."""
    rng = jax.random.PRNGKey(config.seed)

    # initialize the dataset
    logging.rank_zero_info("Building dataset...")
    rng, data_rng = jax.random.split(rng, num=2)
    p_datamodule = fdl.build(config.data)
    datamodule = p_datamodule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        deterministic=config.deterministic,
        drop_remainder=config.drop_remainder,
        rng=data_rng,
    )
    assert isinstance(datamodule, _mixin.DataMixin), (
        "The datamodule must be an instance of `DataMixin`, "
        f"but got {type(datamodule)} instead."
    )
    logging.rank_zero_info(
        "Dataset %s built.",
        datamodule.__class__.__name__,
    )

    # initialize the model
    logging.rank_zero_info("Building model...")
    rng, init_rng = jax.random.split(rng, num=2)
    model = fdl.build(config.model)
    assert isinstance(model, _mixin.ModelMixin), (
        "The model must be an instance of `ModelMixin`, "
        f"but got {type(model)} instead."
    )
    params = model.init(rngs=init_rng)
    logging.rank_zero_info("Model %s built.", model.__class__.__name__)

    # initialize the train state
    logging.rank_zero_info("Building train state...")
    learning_rate_scheduler = fdl.build(config.lr_scheduler)
    p_tx = fdl.build(config.optimizer)
    tx = p_tx(learning_rate=learning_rate_scheduler)
    if config.grad_clip_method == "norm":
        tx = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_value or 1.0),
            tx,
        )
    elif config.grad_clip_method == "value":
        tx = optax.chain(
            optax.clip(config.grad_clip_value or 1.0),
            tx,
        )
    assert isinstance(tx, optax.GradientTransformation)
    state = TrainState.create(
        model=model,
        params=params,
        tx=tx,
        ema_rate=config.ema_rate,
    )
    logging.rank_zero_info("Train state %s built.", state.__class__.__name__)

    # prepare the training step
    rng, train_rng = jax.random.split(rng, num=2)
    p_training_step = functools.partial(training_step, train_rng)
    p_training_step = functools.partial(p_training_step, model)
    p_training_step = jax.pmap(p_training_step, axis_name="batch")

    log_dir = os.path.join(
        work_dir,
        config.name,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    # checkpointing
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    checkpoint_mngr = checkpoint.MultihostCheckpoint(
        multihost_base_directory=ckpt_dir,
        max_to_keep=2,
    )
    if config.checkpoint_dir:
        # TODO (juanwulu): Implement checkpoint resume functionality
        raise NotImplementedError("Checkpoint resume not implemented yet.")

    # prepare the metric writer
    writer = metric_writers.create_default_writer(
        logdir=log_dir,
        asynchronous=False,
    )
    # TODO (juanwulu): resolve the issue with `hparams`
    # writer.write_hparams(dataclasses.asdict(config))
    callbacks = []
    progress_report = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps,
        writer=writer,
    )
    if jax.process_index() == 0:
        # ensure only executing callbacks on rank zero
        callbacks.append(progress_report)
        if config.profile:
            callbacks.append(
                periodic_actions.Profile(
                    logdir=log_dir,
                    num_profile_steps=5,
                )
            )

    step = state.step
    state = jax_utils.replicate(state)
    train_metrics = collections.defaultdict(list)
    logging.rank_zero_info("=========== Training initiated ===========")
    with metric_writers.ensure_flushes(writer):
        try:
            while True:
                for batch in datamodule.train_dataloader():
                    batch = shard(batch)
                    with jax.profiler.StepTraceAnnotation(
                        "train", step_num=step
                    ):
                        state, outputs = p_training_step(state, batch)
                    for k, v in outputs.items():
                        train_metrics[k].append(jax.device_get(v).mean())

                    for cb in callbacks:
                        cb(step)

                    if step % config.log_every_n_steps == 0:
                        lr = learning_rate_scheduler(step)
                        output_args = {
                            f"train/{k.replace('_', ' ')}": sum(v) / len(v)
                            for k, v in train_metrics.items()
                        }
                        output_args = dict(lr=lr) | output_args
                        writer.write_scalars(
                            step=step + 1, scalars=output_args
                        )
                    step += 1

                    if step % config.check_val_every_n_steps == 0:
                        logging.rank_zero_info("Running evaluation...")
                        eval_metrics = collections.defaultdict(list)
                        for batch in datamodule.val_dataloader():
                            batch = shard(batch)
                            _, outputs = p_training_step(state, batch)
                            for k, v in outputs.items():
                                eval_metrics[k].append(
                                    jax.device_get(v).mean()
                                )
                        output_args = {
                            f"eval/{k.replace('_', ' ')}": sum(v) / len(v)
                            for k, v in eval_metrics.items()
                        }
                        writer.write_scalars(step=step, scalars=output_args)
                        logging.rank_zero_info(
                            "Evaluation completed. Saving..."
                        )
                        with progress_report.timed("checkpointing"):
                            filepath = checkpoint_mngr.save(
                                state=jax_utils.unreplicate(state)
                            )
                        logging.rank_zero_info(
                            "Checkpoint saved to %s",
                            filepath,
                        )

                if step > config.num_train_steps:
                    break
        except Exception as e:
            logging.rank_zero_info("Exception occurred during training: %s", e)
            raise e
        finally:
            state = jax_utils.unreplicate(state)
            logging.rank_zero_info(
                "Training finished. Final step: %d",
                state.step,
            )
