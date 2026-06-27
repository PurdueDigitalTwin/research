"""Evaluate bias-corrected beta*(t) using an independent v_ref.

Loads the MeanFlow baseline (beta=0) and the converged unconditional v_ref,
computes the ratio-based beta*(t) at each t, and runs a circularity
cross-check comparing the independent v_ref to the EMA-proxy estimate.

Usage::

    bazelisk run --config=tpu \
        //src/projects/generative/vamf/experiments:eval_betastar_vref -- \
        --mf_checkpoint_dir=gs://.../<step> \
        --vref_checkpoint_dir=gs://.../<step> \
        --output_path=docs/generative/vamf/betastar_vref_results.json
"""
import json
import os
import typing

from absl import app
from absl import flags
from etils import epath
import fiddle as fdl
import jax
from jax import numpy as jnp
from jax import random as jrnd
import numpy as np
from orbax import checkpoint as ocp

from src.projects.generative import config as _cfg
from src.projects.generative.vamf.scripts import betastar_from_vref as _bs
from src.utilities import logging as _logging

flags.DEFINE_string(
    "mf_checkpoint_dir", None,
    "MeanFlow baseline checkpoint step dir (e.g. .../240000).",
    required=True,
)
flags.DEFINE_string(
    "vref_checkpoint_dir", None,
    "Converged unconditional v_ref checkpoint step dir (e.g. .../55000).",
    required=True,
)
flags.DEFINE_string(
    "mf_config_fn", "meanflow_dit_imagenet_256_latent",
    "Config function for the MeanFlow baseline.",
)
flags.DEFINE_string(
    "vref_config_fn", "fm_dit_imagenet_256_latent_uncond",
    "Config function for the v_ref model.",
)
flags.DEFINE_string(
    "output_path", None, "JSON output path.", required=True,
)
flags.DEFINE_float("beta_no_bias", 0.94, "Paper's matrix-form bound.")
flags.DEFINE_integer("n_batches", 16, "Batches per t (batch_size=64).")
flags.DEFINE_integer("pool_size", 2048, "Latent pool size.")
flags.DEFINE_integer("seed", 42, "Random seed.")


def _restore_ema_params(model, checkpoint_dir: str) -> typing.Any:
    """Restore EMA params from <checkpoint_dir>/params/."""
    init_rng = jrnd.PRNGKey(0)
    params, _ = model.init(batch=None, rngs=init_rng)
    params_dir = epath.Path(
        os.path.join(checkpoint_dir.rstrip("/"), "params")
    )
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(sharding=sharding), params,
    )
    handler = ocp.PyTreeCheckpointHandler()
    return handler.restore(
        directory=params_dir, item=params,
        transforms=None, restore_args=restore_args,
    )


def _restore_online_params(model, checkpoint_dir: str) -> typing.Any:
    """Restore online (non-EMA) params from <checkpoint_dir>/state/."""
    from src.core import train_state as _ts
    import optax
    init_rng = jrnd.PRNGKey(0)
    params, _ = model.init(batch=None, rngs=init_rng)
    state = _ts.TrainState.create(
        params=params, tx=optax.adam(1e-4), ema_rate=0.9999,
    )
    state_template = state.replace(ema_params={})
    state_dir = epath.Path(
        os.path.join(checkpoint_dir.rstrip("/"), "state")
    )
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(sharding=sharding), state_template,
    )
    handler = ocp.PyTreeCheckpointHandler()
    restored = handler.restore(
        directory=state_dir, item=state_template,
        transforms=None, restore_args=restore_args,
    )
    return restored.params


def _build_apply_fn(model, params):
    """Build a (z, t) -> velocity closure at null class, boundary r=t."""
    null_label = jnp.int32(model.num_classes)

    @jax.jit
    def apply_fn(z, t_scalar):
        n = z.shape[0]
        t = jnp.full((n,), t_scalar, dtype=jnp.float32)
        timestamps = model._make_timestamps(t_in=t, r_in=t)
        labels = jnp.full((n,), null_label, dtype=jnp.int32)
        return model._network.apply(
            variables={"params": params},
            inputs=z, timestamps=timestamps, labels=labels,
            edm_cond=None, deterministic=True,
        )

    return apply_fn


def _build_latent_pool(config_fn_name, pool_size, seed):
    """Load real ImageNet latents into a numpy pool."""
    config_fn = getattr(_cfg, config_fn_name)
    exp_config = config_fn()
    data_kwargs = {
        "batch_size": 64, "num_workers": 1,
        "deterministic": True, "drop_remainder": True,
    }
    datamodule = fdl.build(exp_config.data.module)(**data_kwargs)
    if hasattr(datamodule, "setup"):
        datamodule.setup()

    latents = []
    n_collected = 0
    it = iter(datamodule.train_dataloader())
    while n_collected < pool_size:
        batch = next(it)
        if "latent_mean" in batch:
            mean = np.asarray(batch["latent_mean"], dtype=np.float32)
            logvar = np.asarray(batch["latent_logvar"], dtype=np.float32)
            std = np.exp(0.5 * logvar)
            rng = np.random.default_rng(seed + n_collected)
            noise = rng.normal(size=mean.shape).astype(np.float32)
            z = (mean + std * noise) * 0.18215
        elif "latent" in batch:
            z = np.asarray(batch["latent"], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected batch keys: {list(batch.keys())}")
        latents.append(z)
        n_collected += z.shape[0]
    pool = np.concatenate(latents, axis=0)[:pool_size]
    _logging.rank_zero_info("Latent pool: %s", pool.shape)
    return pool


def main(argv):
    del argv
    F = flags.FLAGS

    _logging.rank_zero_info("=" * 60)
    _logging.rank_zero_info("Step 3: bias-corrected beta*(t) via independent v_ref")
    _logging.rank_zero_info("MF checkpoint: %s", F.mf_checkpoint_dir)
    _logging.rank_zero_info("v_ref checkpoint: %s", F.vref_checkpoint_dir)
    _logging.rank_zero_info("beta_no_bias: %s", F.beta_no_bias)
    _logging.rank_zero_info("=" * 60)

    # --- Build models ---
    mf_config = getattr(_cfg, F.mf_config_fn)()
    mf_model = fdl.build(mf_config.model)()
    vref_config = getattr(_cfg, F.vref_config_fn)()
    vref_model = fdl.build(vref_config.model)()

    # --- Load EMA params ---
    _logging.rank_zero_info("Loading MeanFlow EMA params...")
    mf_ema_params = _restore_ema_params(mf_model, F.mf_checkpoint_dir)
    _logging.rank_zero_info("Loading v_ref EMA params...")
    vref_ema_params = _restore_ema_params(vref_model, F.vref_checkpoint_dir)

    # --- Load MF online params for circularity cross-check ---
    _logging.rank_zero_info("Loading MeanFlow online params (for EMA-proxy cross-check)...")
    mf_online_params = _restore_online_params(mf_model, F.mf_checkpoint_dir)

    # --- Build apply closures ---
    mf_apply_fn = _build_apply_fn(mf_model, mf_ema_params)
    vref_apply_fn = _build_apply_fn(vref_model, vref_ema_params)
    mf_online_apply_fn = _build_apply_fn(mf_model, mf_online_params)

    # --- Load latent pool ---
    _logging.rank_zero_info("Loading ImageNet latent pool...")
    pool = _build_latent_pool(F.mf_config_fn, F.pool_size, F.seed)
    rng = np.random.default_rng(F.seed)
    batch_size = 64

    def get_latent_batch():
        idx = rng.integers(0, pool.shape[0], batch_size)
        return pool[idx]

    # --- Wrappers: numpy-in, numpy-out ---
    def mf_apply(xt, t):
        return np.asarray(mf_apply_fn(jnp.asarray(xt), t))

    def vref_apply(xt, t):
        return np.asarray(vref_apply_fn(jnp.asarray(xt), t))

    def mf_online_apply(xt, t):
        return np.asarray(mf_online_apply_fn(jnp.asarray(xt), t))

    t_grid = [0.1, 0.3, 0.5, 0.7, 0.9]

    # --- Main eval: MF-EMA vs independent v_ref ---
    _logging.rank_zero_info("=" * 60)
    _logging.rank_zero_info("MAIN: MF-EMA boundary vs independent v_ref")
    _logging.rank_zero_info("=" * 60)
    per_t_main = _bs.run_eval(
        get_latent_batch, mf_apply, vref_apply, t_grid,
        n_batches=F.n_batches, beta_no_bias=F.beta_no_bias, seed=F.seed,
    )
    _bs.print_table(per_t_main, F.beta_no_bias)

    # --- Cross-check (a): noise(t) profile sanity ---
    _logging.rank_zero_info("\n=== Cross-check (a): noise(t) = E||v_cond - v_ref||^2 ===")
    _logging.rank_zero_info("Expected: ~d=4096 at t->0, ~E||x0||^2=2799 at t->1, peak ~t=0.5")
    for t in sorted(per_t_main):
        s = per_t_main[t]
        _logging.rank_zero_info(
            "  t=%.1f: noise=%.1f (cv=%.3f)", t, s["noise_mean"], s["noise_cv"]
        )

    # --- Cross-check (b): EMA-proxy vs v_ref (circularity check) ---
    _logging.rank_zero_info("\n" + "=" * 60)
    _logging.rank_zero_info("CIRCULARITY CHECK: MF-online boundary vs MF-EMA boundary")
    _logging.rank_zero_info("  (paper's self-referential proxy: bias = ||u_online - u_ema||^2)")
    _logging.rank_zero_info("=" * 60)

    rng_circ = np.random.default_rng(F.seed + 1000)

    def get_batch_circ():
        idx = rng_circ.integers(0, pool.shape[0], batch_size)
        return pool[idx]

    per_t_circ = {}
    for t in t_grid:
        bias_all, noise_all = [], []
        for _ in range(F.n_batches):
            x0 = np.asarray(get_batch_circ(), dtype=float)
            e = rng_circ.standard_normal(x0.shape)
            xt = (1.0 - t) * x0 + t * e
            v_cond = e - x0
            u_online = np.asarray(mf_online_apply(xt, t), dtype=float)
            u_ema = np.asarray(mf_apply(xt, t), dtype=float)
            v_ref = np.asarray(vref_apply(xt, t), dtype=float)
            ax = tuple(range(1, x0.ndim))
            bias_all.append(np.sum((u_online - u_ema) ** 2, axis=ax))
            noise_all.append(np.sum((v_cond - v_ref) ** 2, axis=ax))
        per_t_circ[t] = _bs.ratio_stats(
            np.concatenate(bias_all), np.concatenate(noise_all),
            beta_no_bias=F.beta_no_bias,
        )

    _logging.rank_zero_info("\n=== EMA-proxy ratio (circularity check) ===")
    _bs.print_table(per_t_circ, F.beta_no_bias)

    # --- Comparison table ---
    _logging.rank_zero_info("\n=== COMPARISON: ratio_vref vs ratio_emaproxy per t ===")
    _logging.rank_zero_info(f"{'t':>6}{'ratio_vref':>12}{'ratio_ema':>12}{'agree?':>10}")
    for t in sorted(t_grid):
        rv = per_t_main[t]["ratio"]
        re = per_t_circ[t]["ratio"]
        agree = "YES" if abs(rv - re) / max(rv, re, 1e-9) < 0.25 else "NO"
        _logging.rank_zero_info(f"  {t:.1f}{rv:>12.3f}{re:>12.3f}{agree:>10}")

    # --- Save JSON ---
    if jax.process_index() == 0:
        payload = {
            "metadata": {
                "mf_checkpoint": F.mf_checkpoint_dir,
                "vref_checkpoint": F.vref_checkpoint_dir,
                "beta_no_bias": F.beta_no_bias,
                "n_batches": F.n_batches,
                "batch_size": batch_size,
                "pool_size": F.pool_size,
                "t_grid": t_grid,
            },
            "main_vref": {
                str(t): {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in per_t_main[t].items()
                }
                for t in per_t_main
            },
            "circularity_emaproxy": {
                str(t): {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in per_t_circ[t].items()
                }
                for t in per_t_circ
            },
        }
        out = epath.Path(F.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        _logging.rank_zero_info("Results saved to %s", out)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
