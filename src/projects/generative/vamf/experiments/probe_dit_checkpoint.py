"""Variance-amplification diagnostic for DiT MeanFlow checkpoints.

Mirrors ``run_diagnostic_unet.py`` (CIFAR-10 / U-Net) but operates on
ImageNet-256 *latents* and the class-conditional DiT-B/4 backbone. The
probe restores params + EMA params from an Orbax checkpoint, builds a
plain ``u_fn(z, r, t) -> u`` closure, and runs the same Theorem-1
``variance_amplification`` primitive used for the toy and CIFAR-10
results. Output JSON has the same shape so the headline NR figure
script can ingest both U-Net and DiT runs uniformly.

Usage::

    bazelisk run --config=tpu \\
        //src/projects/generative/vamf/experiments:probe_dit_checkpoint -- \\
        --checkpoint_dir=gs://pdt_training/juanwu/meanflow/<exp>/checkpoints/<step> \\
        --config_fn=meanflow_dit_imagenet_256_latent \\
        --output_path=docs/generative/vamf/results/dit_probe_<exp>_step<step>.json
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
from src.projects.generative.vamf.model import diagnostic as _diagnostic
from src.utilities import logging as _logging

# ==============================================================================
# Flags
# ==============================================================================
flags.DEFINE_string(
    name="checkpoint_dir",
    default=None,
    required=True,
    help=(
        "Path to a single checkpoint step directory, e.g. "
        "gs://pdt_training/juanwu/meanflow/<exp>/checkpoints/<step>."
    ),
)
flags.DEFINE_string(
    name="config_fn",
    default=None,
    required=True,
    help=(
        "Name of the Fiddle config function in "
        "src.projects.generative.config that produced this checkpoint, "
        "e.g. meanflow_dit_imagenet_256_latent."
    ),
)
flags.DEFINE_string(
    name="output_path",
    default=None,
    required=True,
    help="Where to write the JSON output (variance ratios + raw vars).",
)
flags.DEFINE_integer(
    name="n_samples",
    default=512,
    help=(
        "Per-(t,r) sample budget for variance amplification. DiT is "
        "16x heavier than U-Net per forward; default is correspondingly "
        "smaller than the U-Net script."
    ),
)
flags.DEFINE_integer(
    name="n_probes",
    default=5,
    help="Hutchinson probes per Jacobian-norm evaluation.",
)
flags.DEFINE_float(
    name="diag_gap",
    default=0.25,
    help="Fixed (t-r) for variance-amplification + Jacobian-norm.",
)
flags.DEFINE_list(
    name="experiments",
    default=["1"],
    help="Experiments (subset of '1','2','4'). DiT defaults to Exp 1 only.",
)
flags.DEFINE_integer(
    name="seed",
    default=42,
    help="Random seed for reproducibility.",
)


def _restore_params(model, checkpoint_dir: str) -> typing.Any:
    """Restore EMA params from ``<checkpoint_dir>/params``."""
    _logging.rank_zero_info("Initializing model for param structure...")
    init_rng = jrnd.PRNGKey(0)
    params, _ = model.init(batch=None, rngs=init_rng)

    params_dir = epath.Path(os.path.join(checkpoint_dir.rstrip("/"), "params"))
    _logging.rank_zero_info("Loading checkpoint from %s...", params_dir)
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(sharding=sharding),
        params,
    )
    handler = ocp.PyTreeCheckpointHandler()
    params = handler.restore(
        directory=params_dir,
        item=params,
        transforms=None,
        restore_args=restore_args,
    )
    _logging.rank_zero_info("Checkpoint loaded.")
    return params


def _build_u_fn(model, params):
    """Wrap the DiT backbone as a ``u_fn(z, r, t) -> u`` closure.

    Uses the unconditional / null class label (``num_classes`` is the
    drop index by convention in the DiT class embedder). This isolates
    Theorem 1 from class-conditioning effects: the variance ratio we
    care about is intrinsic to the (z, r, t) sweep.
    """
    null_label = jnp.int32(model.num_classes)

    def u_fn(z, r, t):
        n = z.shape[0]
        timestamps = model._make_timestamps(t_in=t, r_in=r)
        labels = jnp.full((n,), null_label, dtype=jnp.int32)
        out = model._network.apply(
            variables={"params": params},
            inputs=z,
            timestamps=timestamps,
            labels=labels,
            edm_cond=None,
            deterministic=True,
        )
        assert isinstance(out, jax.Array)
        return out

    return u_fn


def _build_latent_sampler(
    config_fn_name: str,
    pool_size: int = 1024,
    seed: int = 42,
) -> typing.Callable[[jax.Array, int], jax.Array]:
    """Sample ImageNet latents (32x32x4) from a small cached pool.

    Reads a single shard of the pre-encoded TFDS-style latent dataset
    using the data module from the Fiddle config, materializes the first
    ``pool_size`` examples on host, and returns a ``(rng, n) -> batch``
    sampler over them.
    """
    _logging.rank_zero_info(
        "Building latent sampler from config %s ...", config_fn_name
    )
    config_fn = getattr(_cfg, config_fn_name)
    exp_config = config_fn()

    # Build the data module via Fiddle. We override the batch size and
    # disable workers so the iterator is deterministic and lightweight.
    data_kwargs = {
        "batch_size": 64,
        "num_workers": 1,
        "deterministic": True,
        "drop_remainder": True,
    }
    datamodule = fdl.build(exp_config.data.module)(**data_kwargs)
    if hasattr(datamodule, "setup"):
        datamodule.setup()

    # Pull batches until we have ``pool_size`` latents.
    latents: typing.List[np.ndarray] = []
    n_collected = 0
    iterator = iter(datamodule.train_dataloader())
    while n_collected < pool_size:
        batch = next(iterator)
        if "latent_mean" in batch:
            mean = np.asarray(batch["latent_mean"], dtype=np.float32)
            logvar = np.asarray(batch["latent_logvar"], dtype=np.float32)
            std = np.exp(0.5 * logvar)
            rng = np.random.default_rng(seed + n_collected)
            noise = rng.normal(size=mean.shape).astype(np.float32)
            z = (mean + std * noise) * 0.18215  # SD VAE scaling factor
        elif "latent" in batch:
            z = np.asarray(batch["latent"], dtype=np.float32)
        else:
            raise ValueError(
                f"Unexpected batch keys: {list(batch.keys())} -- "
                "the probe script expects pre-encoded latents."
            )
        latents.append(z)
        n_collected += z.shape[0]
    pool = jnp.asarray(np.concatenate(latents, axis=0)[:pool_size])
    _logging.rank_zero_info(
        "Loaded latent pool of shape %s", tuple(pool.shape)
    )

    def sample(key: jax.Array, n: int) -> jax.Array:
        idx = jrnd.randint(key, shape=(n,), minval=0, maxval=pool.shape[0])
        return pool[idx]

    return sample


def main(argv: typing.List[str]) -> int:
    """Run variance-amplification + optional Jacobian-norm probes."""
    del argv
    F = flags.FLAGS

    _logging.rank_zero_info("=" * 60)
    _logging.rank_zero_info("VaMF DiT Probe (variance amplification)")
    _logging.rank_zero_info("Checkpoint: %s", F.checkpoint_dir)
    _logging.rank_zero_info("Config: %s", F.config_fn)
    _logging.rank_zero_info("Platform: %s", jax.default_backend())
    _logging.rank_zero_info("Devices: %s", jax.devices())
    _logging.rank_zero_info("=" * 60)

    config_fn = getattr(_cfg, F.config_fn)
    exp_config = config_fn()
    model = fdl.build(exp_config.model)()

    params = _restore_params(model, F.checkpoint_dir)
    u_fn = _build_u_fn(model, params)
    sample_x0 = _build_latent_sampler(F.config_fn, seed=F.seed)

    experiments = {int(x) for x in F.experiments}
    t_values = (0.1, 0.3, 0.5, 0.7, 0.9)
    all_results: typing.Dict[str, typing.Any] = {}

    if 1 in experiments:
        _logging.rank_zero_info("=== Experiment 1: Variance Amplification ===")
        key = jrnd.PRNGKey(F.seed)
        all_results[
            "exp1_variance_amplification"
        ] = _diagnostic.variance_amplification(
            u_fn,
            sample_x0,
            key,
            F.n_samples,
            t_probes=t_values,
            fixed_gap=F.diag_gap,
            log_fn=_logging.rank_zero_info,
        )

    if 2 in experiments:
        _logging.rank_zero_info("=== Experiment 2: Curvature Gap ===")
        key = jrnd.PRNGKey(F.seed + 1)
        all_results["exp2_curvature_gap"] = _diagnostic.curvature_gap(
            u_fn,
            sample_x0,
            key,
            F.n_samples,
            t_probes=(0.3, 0.5, 0.7, 0.9),
            log_fn=_logging.rank_zero_info,
        )

    if 4 in experiments:
        _logging.rank_zero_info("=== Experiment 4: Jacobian Norm ===")
        key = jrnd.PRNGKey(F.seed + 2)
        all_results["exp4_jacobian_norm"] = _diagnostic.jacobian_norm(
            u_fn,
            sample_x0,
            key,
            F.n_samples,
            t_probes=t_values,
            fixed_gap=F.diag_gap,
            n_probes=F.n_probes,
            exact=False,  # 32*32*4 = 4096 dims, exact trace infeasible
            log_fn=_logging.rank_zero_info,
        )

    metadata = {
        "checkpoint_dir": F.checkpoint_dir,
        "config_fn": F.config_fn,
        "n_samples": F.n_samples,
        "diag_gap": F.diag_gap,
        "seed": F.seed,
    }
    payload = {"metadata": metadata, "results": all_results}

    output_path = epath.Path(F.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    _logging.rank_zero_info("Results saved to %s", output_path)

    if "exp1_variance_amplification" in all_results:
        _logging.rank_zero_info("=" * 60)
        _logging.rank_zero_info("SUMMARY -- Variance Amplification Ratios")
        _logging.rank_zero_info("=" * 60)
        for k, v in all_results["exp1_variance_amplification"].items():
            _logging.rank_zero_info(
                "  %s: ratio=%.2f  stoch=%.3e  determ=%.3e",
                k,
                v["variance_ratio"],
                v["stochastic_var"],
                v["deterministic_var"],
            )

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
