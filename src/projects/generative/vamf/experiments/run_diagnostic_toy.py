"""Diagnostic experiments for MeanFlow variance amplification analysis.

Trains a vanilla MeanFlow model on a toy 2-D dataset, then probes three
properties of the learned velocity field:

  Exp 1 -- Variance amplification: Var(stochastic loss) / Var(deterministic
           loss) at each t, quantifying the estimator-noise reduction from
           replacing v_cond with u_EMA in the JVP tangent direction.
  Exp 2 -- Curvature gap: ||u(z,r,t) - v_cond||^2 as a function of (t - r),
           showing how the mean-velocity approximation degrades with interval
           length.
  Exp 4 -- Jacobian norm: ||(t-r) J_z u - I||_F vs t, quantifying the
           flow nonlinearity that drives variance amplification.

Outputs a JSON file consumed by ``plot_diagnostics.py``.
"""

import json
import math
import os
import time
import typing

# Use absl.logging directly to avoid tensorflow/wandb deps from
# src.utilities.logging — this script always runs single-process.
from absl import app
from absl import flags
from absl import logging as _logging
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random as jrnd
import numpy as np
import optax

from src.core import train_state as _train_state
from src.projects.generative.vamf.model import diagnostic as _diagnostic


def _rank_zero_info(msg, *args, **kwargs):
    _logging.info(msg, *args, **kwargs)


# ==============================================================================
# Flags
# ==============================================================================
flags.DEFINE_enum(
    name="dataset",
    default="eight_gaussians",
    enum_values=[
        "checkerboard",
        "eight_gaussians",
        "two_moons",
        "swiss_roll",
    ],
    help="Dataset to train on.",
)
flags.DEFINE_integer("hidden_size", 128, "MLP hidden size.")
flags.DEFINE_integer("num_layers", 3, "Number of MLP layers.")
flags.DEFINE_float("ema_rate", 0.999, "EMA decay rate.")
flags.DEFINE_float("overlap_rate", 0.25, "r=t overlap probability.")
flags.DEFINE_float("max_gap", 0.5, "Maximum (t - r).")
flags.DEFINE_integer("train_steps", 200_000, "Training steps.")
flags.DEFINE_integer("batch_size", 256, "Batch size for training.")
flags.DEFINE_float("lr", 0.001, "Learning rate.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("diag_samples", 8192, "Samples per diagnostic probe.")
flags.DEFINE_float("diag_gap", 0.25, "Fixed gap (t - r) for Exp 1 and Exp 4.")
flags.DEFINE_string("work_dir", None, "Output directory.", required=True)
flags.DEFINE_integer(
    "log_every_n_steps", 1000, "Log interval during training."
)


# ==============================================================================
# Datasets (mirrors run_toy.py)
# ==============================================================================
def checkerboard(key: typing.Any, n: int) -> jax.Array:
    x_key, y_key, mask_key = jrnd.split(key, 3)
    x1 = jrnd.uniform(x_key, (n,)) * 4 - 2
    x2 = jrnd.uniform(y_key, (n,)) - (
        2.0 * jrnd.bernoulli(mask_key, 0.5, (n,)).astype(jnp.float32)
    )
    x2 = x2 + (jnp.floor(x1) % 2)
    return jnp.stack([x1, x2], axis=-1) * 2


def eight_gaussians(key: typing.Any, n: int) -> jax.Array:
    z_key, noise_key = jrnd.split(key, num=2)
    angles = jnp.linspace(0, 2 * jnp.pi, 9)[:-1]
    centers = 3.0 * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    idx = jrnd.randint(z_key, (n,), 0, 8)
    noise = jrnd.normal(noise_key, (n, 2)) * 0.25
    return centers[idx] + noise


def two_moons(key: typing.Any, n: int) -> jax.Array:
    k1, k2, k3 = jrnd.split(key, 3)
    n1, n2 = n // 2, n - n // 2
    t1 = jrnd.uniform(k1, (n1,)) * jnp.pi
    t2 = jrnd.uniform(k2, (n2,)) * jnp.pi
    noise = jrnd.normal(k3, (n, 2)) * 0.08
    upper = jnp.stack([jnp.cos(t1), jnp.sin(t1)], axis=-1)
    lower = jnp.stack([1 - jnp.cos(t2), -jnp.sin(t2) + 0.5], axis=-1)
    return (jnp.concatenate([upper, lower], axis=0) + noise) * 2


def swissroll(key: typing.Any, n: int) -> jax.Array:
    k1, k2 = jrnd.split(key)
    t = 1.5 * jnp.pi * (1 + 2 * jrnd.uniform(k1, (n,)))
    noise = jrnd.normal(k2, (n, 2)) * 0.08
    return jnp.stack([t * jnp.cos(t), t * jnp.sin(t)], -1) / 8 + noise


_DATASET_FN = {
    "checkerboard": checkerboard,
    "eight_gaussians": eight_gaussians,
    "two_moons": two_moons,
    "swiss_roll": swissroll,
}


def sample_data(key: jax.Array, dataset: str, n: int) -> jax.Array:
    return _DATASET_FN[dataset](key, n)


# ==============================================================================
# Timestamp sampling (mirrors run_toy.py)
# ==============================================================================
def sample_t_r(
    key: jax.Array,
    shape: typing.Tuple[int, ...],
    overlap_rate: float = 0.25,
    max_gap: float = 0.5,
) -> typing.Tuple[jax.Array, jax.Array]:
    t_key, r_key, mask_key = jrnd.split(key, 3)
    t = jrnd.uniform(t_key, shape, minval=1e-4, maxval=1.0 - 1e-4)
    r = jrnd.uniform(r_key, shape, minval=1e-4, maxval=1.0 - 1e-4)
    t, r = jnp.maximum(t, r), jnp.minimum(t, r)
    r = jnp.maximum(r, t - max_gap)
    mask = jrnd.uniform(mask_key, shape) < overlap_rate
    r = jnp.where(mask, t, r)
    return t, r


# ==============================================================================
# MLP velocity network (mirrors run_toy.py)
# ==============================================================================
class MeanFlowMLP(nn.Module):
    features: int
    num_layers: int

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        r: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        t_embed = self._sinusoidal_embedding(t)
        r_embed = self._sinusoidal_embedding(r)
        out = jnp.concatenate([inputs, t_embed, t_embed - r_embed], axis=-1)
        for i in range(self.num_layers):
            out = nn.Dense(
                features=self.features,
                kernel_init=jax.nn.initializers.variance_scaling(
                    1.0, "fan_avg", "uniform"
                ),
                bias_init=jax.nn.initializers.zeros,
                name=f"fc_{i}",
            )(out)
            out = nn.silu(out)
        out = nn.Dense(
            features=inputs.shape[-1],
            kernel_init=jax.nn.initializers.variance_scaling(
                1e-10, "fan_avg", "uniform"
            ),
            bias_init=jax.nn.initializers.zeros,
            name="fc_out",
        )(out)
        return out

    @staticmethod
    def _sinusoidal_embedding(t: jax.Array, dim: int = 32) -> jax.Array:
        half = dim // 2
        freqs = jnp.exp(
            -math.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / half
        )
        args = t[..., None] * freqs
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


# ==============================================================================
# Training
# ==============================================================================
def train_model(
    dataset: str,
    network: MeanFlowMLP,
    key: jax.Array,
    steps: int,
    batch_size: int,
    lr: float,
    ema_rate: float,
    overlap_rate: float,
    max_gap: float,
    log_every: int,
) -> _train_state.TrainState:
    """Train vanilla MeanFlow and return the final train state."""
    key, init_key = jrnd.split(key)
    variables = network.init(
        init_key,
        jnp.zeros((1, 2)),
        jnp.zeros(1),
        jnp.zeros(1),
    )
    params = {"velocity": variables["params"]}

    lr_schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=steps, alpha=0.01
    )
    state = _train_state.TrainState.create(
        params=params, tx=optax.adam(lr_schedule), ema_rate=ema_rate
    )

    def _step(state, key):
        k_data, k_tr, k_e = jrnd.split(key, 3)
        x0 = sample_data(k_data, dataset, batch_size)

        def loss_fn(params):
            vel_p = params["velocity"]
            t, r = sample_t_r(k_tr, (batch_size,), overlap_rate, max_gap)
            e = jrnd.normal(k_e, x0.shape)
            z = (1.0 - t[:, None]) * x0 + t[:, None] * e
            v_cond = e - x0

            def u_fn(z_, r_, t_):
                return network.apply({"params": vel_p}, z_, t_, r_)

            u, dudt = jax.jvp(
                u_fn,
                (z, r, t),
                (v_cond, jnp.zeros_like(r), jnp.ones_like(t)),
            )
            gap = jnp.clip(t - r, 0.0, 1.0)
            v_pred = u + gap[:, None] * jax.lax.stop_gradient(dudt)
            v_target = jax.lax.stop_gradient(v_cond)
            per_sample = jnp.sum(jnp.square(v_pred - v_target), axis=-1)
            loss = jnp.mean(per_sample)
            return loss, loss

        grads, loss = jax.grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), loss

    train_step = jax.jit(_step)

    _rank_zero_info("Training MeanFlow on %s for %d steps...", dataset, steps)
    t0 = time.time()
    for step in range(steps):
        key, step_key = jrnd.split(key)
        state, loss = train_step(state, step_key)
        if step % log_every == 0 or step == steps - 1:
            _rank_zero_info(
                "[%6d/%d] loss=%.4f (%.1fs)",
                step,
                steps,
                float(loss),
                time.time() - t0,
            )
    _rank_zero_info("Training finished in %.1fs.", time.time() - t0)
    return state


# ==============================================================================
# Main
# ==============================================================================
def main(argv: typing.List[str]) -> int:
    del argv
    FLAGS = flags.FLAGS
    key = jrnd.PRNGKey(FLAGS.seed)

    network = MeanFlowMLP(
        features=FLAGS.hidden_size, num_layers=FLAGS.num_layers
    )

    # ---- Train ---------------------------------------------------------------
    key, train_key = jrnd.split(key)
    state = train_model(
        dataset=FLAGS.dataset,
        network=network,
        key=train_key,
        steps=FLAGS.train_steps,
        batch_size=FLAGS.batch_size,
        lr=FLAGS.lr,
        ema_rate=FLAGS.ema_rate,
        overlap_rate=FLAGS.overlap_rate,
        max_gap=FLAGS.max_gap,
        log_every=FLAGS.log_every_n_steps,
    )
    vel_params = state.ema_params["velocity"]

    # Bind the EMA params into the velocity-field closure expected by the
    # shared diagnostic primitives. The MLP module has __call__(inputs, r, t)
    # but the trace / JVP logic uses (z, r, t) outer convention; we pass the
    # outer t into the module's r slot and vice versa to match run_toy.
    def u_fn(z: jax.Array, r: jax.Array, t: jax.Array) -> jax.Array:
        output = network.apply({"params": vel_params}, z, t, r)
        assert isinstance(output, jax.Array)

        return output

    def sample_x0(rng, n):
        return sample_data(rng, FLAGS.dataset, n)

    # ---- Diagnostics ---------------------------------------------------------
    diag_data: typing.Dict[str, typing.Any] = {}

    _rank_zero_info("Running Experiment 1 (variance amplification)...")
    key, k = jrnd.split(key)
    diag_data[
        "exp1_variance_amplification"
    ] = _diagnostic.variance_amplification(
        u_fn,
        sample_x0,
        k,
        FLAGS.diag_samples,
        fixed_gap=FLAGS.diag_gap,
        log_fn=_rank_zero_info,
    )

    _rank_zero_info("Running Experiment 2 (curvature gap)...")
    key, k = jrnd.split(key)
    diag_data["exp2_curvature_gap"] = _diagnostic.curvature_gap(
        u_fn,
        sample_x0,
        k,
        FLAGS.diag_samples,
        log_fn=_rank_zero_info,
    )

    _rank_zero_info("Running Experiment 4 (Jacobian norm)...")
    key, k = jrnd.split(key)
    diag_data["exp4_jacobian_norm"] = _diagnostic.jacobian_norm(
        u_fn,
        sample_x0,
        k,
        FLAGS.diag_samples,
        fixed_gap=FLAGS.diag_gap,
        exact=True,  # 2-D toy: enumerate the standard basis exactly.
        log_fn=_rank_zero_info,
    )

    # ---- Save ----------------------------------------------------------------
    os.makedirs(FLAGS.work_dir, exist_ok=True)
    out_path = os.path.join(
        FLAGS.work_dir,
        f"diagnostics_{FLAGS.dataset}_{FLAGS.seed}.json",
    )
    with open(out_path, "w") as f:
        json.dump(diag_data, f, indent=2)
    _rank_zero_info("Saved diagnostics to %s", out_path)

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
