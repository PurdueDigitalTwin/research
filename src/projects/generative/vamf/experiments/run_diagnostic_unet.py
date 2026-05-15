"""Diagnostic measurements for the MeanFlow U-Net (CIFAR-10 default).

Usage::

    bazelisk run --config=cuda \\
        //src/projects/generative/vamf/experiments:run_diagnostic_unet -- \\
        --checkpoint_dir="$CHECKPOINT_DIR" \\
        --output_dir=docs/generative/vamf/results
"""

import json
import math
import os
import typing

from absl import app
from absl import flags
import datasets
from etils import epath
import jax
from jax import numpy as jnp
from jax import random as jrnd
import jaxtyping
import numpy as np
from orbax import checkpoint as ocp

from src.projects.generative.meanflow import MeanFlowUNetModel
from src.projects.generative.vamf.model import diagnostic as _diagnostic
from src.utilities import logging as _logging

# ==============================================================================
# Flags
# ==============================================================================
flags.DEFINE_string(
    name="checkpoint_dir",
    default=None,
    required=True,
    help="Path to Orbax checkpoint (GCS or local).",
)
flags.DEFINE_string(
    name="output_dir",
    default="docs/generative/vamf/results",
    help="Directory to save results.",
)
flags.DEFINE_integer(
    name="n_samples",
    default=3200,
    help="Total samples per diagnostic probe.",
)
flags.DEFINE_integer(
    name="n_probes",
    default=5,
    help="Hutchinson probes per Jacobian-norm evaluation.",
)
flags.DEFINE_float(
    name="diag_gap",
    default=0.0,
    help=(
        "Fixed (t-r) for variance-amplification and Jacobian-norm "
        "experiments. The historical CIFAR-10 sweep used 0.0; pass "
        "a positive value for parity with the toy MLP runner."
    ),
)
flags.DEFINE_list(
    name="experiments",
    default=["1", "2", "4"],
    help="Experiment numbers to run (comma-separated subset of 1, 2, 4).",
)
flags.DEFINE_integer(
    name="seed",
    default=42,
    help="Random seed for reproducibility.",
)


# ----- Model + checkpoint loader --------------------------------------------
def build_model() -> MeanFlowUNetModel:
    r"""Build the MeanFlow U-Net matching the training config."""
    return MeanFlowUNetModel(
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
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=None,
    )


def load_params(
    model: MeanFlowUNetModel,
    checkpoint_dir: str,
) -> jaxtyping.PyTree:
    """Load EMA params from an Orbax checkpoint's ``params/`` subdirectory."""
    _logging.rank_zero_info("Initializing model for param structure...")
    init_rng = jrnd.PRNGKey(0)
    params, _ = model.init(batch=None, rngs=init_rng)

    params_dir = epath.Path(os.path.join(checkpoint_dir.rstrip("/"), "params"))
    _logging.rank_zero_info("Loading checkpoint from %s...", params_dir)

    # Single-device sharding so a multi-process checkpoint loads on one host.
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


# ----- Data sampler ----------------------------------------------------------
def build_cifar10_sampler(
    n_pool_batches: int = 50,
    pool_batch_size: int = 64,
    seed: int = 42,
) -> typing.Callable[[jax.Array, int], jax.Array]:
    r"""Return ``sample_x0(rng, n)`` drawing from a fixed image pool."""

    # NOTE: We pre-load a pool of CIFAR-10 images (in [-1, 1]) and sample
    # uniformly from it on each call. This matches the original diagnostic's
    # behaviour while exposing the standard ``(rng, n) -> batch`` interface
    # that the shared diagnostic primitives expect.
    _logging.rank_zero_info("Loading CIFAR-10...")
    ds = datasets.load_dataset(
        "uoft-cs/cifar10",
        split="train",
        token=os.getenv("HF_TOKEN", None),
        revision="0b2714987fa478483af9968de7c934580d0bb9a2",
    )
    images = np.stack([np.array(item["img"]) for item in ds])  # type: ignore
    images = images.astype(np.float32) / 255.0
    images = images * 2.0 - 1.0  # to [-1, 1]
    pool_size = n_pool_batches * pool_batch_size
    rng = np.random.default_rng(seed)
    pool_idx = rng.choice(len(images), size=pool_size, replace=False)
    pool = jnp.asarray(images[pool_idx])
    _logging.rank_zero_info("Loaded pool of %d CIFAR-10 images.", pool_size)

    def sample(key: jax.Array, n: int) -> jax.Array:
        """Uniformly sample n images from the pool (with replacement)."""
        idx = jrnd.randint(key, shape=(n,), minval=0, maxval=pool.shape[0])
        return pool[idx]

    return sample


# ----- u_fn closure ----------------------------------------------------------
def build_u_fn(
    model: MeanFlowUNetModel,
    params: jaxtyping.PyTree,
) -> typing.Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Return a velocity-field closure ``u_fn(z, r, t) -> u``."""

    def u_fn(z: jax.Array, r: jax.Array, t: jax.Array) -> jax.Array:
        timestamps = model._make_timestamps(t_in=t, r_in=r)
        output = model._network.apply(
            variables={"params": params},
            inputs=z,
            timestamps=timestamps,
            edm_cond=None,
            deterministic=True,
        )
        assert isinstance(output, jax.Array)

        return output

    return u_fn


# ----- Main ------------------------------------------------------------------
def main(argv: typing.List[str]) -> int:
    del argv  # unused
    FLAGS = flags.FLAGS

    _logging.rank_zero_info("=" * 60)
    _logging.rank_zero_info("VaMF Diagnostic Measurements")
    _logging.rank_zero_info("Checkpoint: %s", FLAGS.checkpoint_dir)
    _logging.rank_zero_info("Platform: %s", jax.default_backend())
    _logging.rank_zero_info("Devices: %s", jax.devices())
    _logging.rank_zero_info("=" * 60)

    model = build_model()
    params = load_params(model, FLAGS.checkpoint_dir)
    u_fn = build_u_fn(model, params)
    sample_x0 = build_cifar10_sampler(seed=FLAGS.seed)

    experiments = {int(x) for x in FLAGS.experiments}
    t_values = (0.1, 0.3, 0.5, 0.7, 0.9)
    all_results: typing.Dict[str, typing.Any] = {}

    if 1 in experiments:
        _logging.rank_zero_info("=== Experiment 1: Variance Amplification ===")
        key = jrnd.PRNGKey(FLAGS.seed)
        all_results[
            "exp1_variance_amplification"
        ] = _diagnostic.variance_amplification(
            u_fn,
            sample_x0,
            key,
            FLAGS.n_samples,
            t_probes=t_values,
            fixed_gap=FLAGS.diag_gap,
            log_fn=_logging.rank_zero_info,
        )

    if 2 in experiments:
        _logging.rank_zero_info("=== Experiment 2: Curvature Gap ===")
        key = jrnd.PRNGKey(FLAGS.seed + 1)
        all_results["exp2_curvature_gap"] = _diagnostic.curvature_gap(
            u_fn,
            sample_x0,
            key,
            FLAGS.n_samples,
            t_probes=(0.3, 0.5, 0.7, 0.9),
            log_fn=_logging.rank_zero_info,
        )

    if 4 in experiments:
        _logging.rank_zero_info("=== Experiment 4: Jacobian Norm ===")
        key = jrnd.PRNGKey(FLAGS.seed + 2)
        all_results["exp4_jacobian_norm"] = _diagnostic.jacobian_norm(
            u_fn,
            sample_x0,
            key,
            FLAGS.n_samples,
            t_probes=t_values,
            fixed_gap=FLAGS.diag_gap,
            n_probes=FLAGS.n_probes,
            exact=False,  # CIFAR-10: 3072 dims, exact trace infeasible.
            log_fn=_logging.rank_zero_info,
        )

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    output_path = os.path.join(FLAGS.output_dir, "diagnostic_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    _logging.rank_zero_info("Results saved to %s", output_path)

    # Brief summary.
    _logging.rank_zero_info("=" * 60)
    _logging.rank_zero_info("SUMMARY")
    _logging.rank_zero_info("=" * 60)
    if "exp1_variance_amplification" in all_results:
        _logging.rank_zero_info("Exp 1 -- Variance Amplification Ratios:")
        for k, v in all_results["exp1_variance_amplification"].items():
            _logging.rank_zero_info(
                "  %s: ratio = %.2f (stoch=%.1f, determ=%.1f)",
                k,
                v["variance_ratio"],
                v["stochastic_var"],
                v["deterministic_var"],
            )
    if "exp4_jacobian_norm" in all_results:
        _logging.rank_zero_info("Exp 4 -- Jacobian Factor ||J||:")
        for k, v in all_results["exp4_jacobian_norm"].items():
            _logging.rank_zero_info(
                "  %s: ||J||_F ~ %.1f",
                k,
                v["J_norm_mean"],
            )

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
