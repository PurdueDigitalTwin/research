"""Diagnostic measurements for MeanFlow variance analysis.

Implements Experiments 1, 2, and 4 from the VaMF diagnostic plan:
  1. Jacobian variance amplification (Theorem 1)
  2. Curvature gap vs interval length (Theorem 3)
  4. Jacobian norm growth

Usage (on TPU VM):
    python src/projects/generative/tools/diagnostic.py \
        --checkpoint_dir gs://pdt_gen_ai/juanwu/meanflow/meanflow_unet_cifar_10_20260412_191003/checkpoints/800000 \
        --output_dir docs/generative/vamf/results
"""

import argparse
import json
import math
import os
import typing

import datasets
from etils import epath
import jax
from jax import numpy as jnp
import jaxtyping
import numpy as np
from orbax import checkpoint as ocp

from src.projects.generative.meanflow import MeanFlowUNetModel


def build_model() -> MeanFlowUNetModel:
    """Build the MeanFlow U-Net model matching the training config."""
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
            mean=-0.6, stddev=1.6, r_mean=-4.0, r_stddev=1.6
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
    """Load model params (EMA) from an Orbax checkpoint.

    The checkpoint is saved by CheckpointManager with two items:
    ``state`` (TrainState without ema_params) and ``params``
    (the EMA parameters). We restore only the ``params`` item.

    Args:
        model: MeanFlow model (used to get param structure).
        checkpoint_dir: Path to the step directory, e.g.
            ``.../checkpoints/800000/``. Inside it, params are
            stored in the ``params/`` subdirectory.
    """
    print("Initializing model for param structure...", flush=True)
    init_rng = jax.random.PRNGKey(0)
    params, _ = model.init(batch=None, rngs=init_rng)

    params_dir = epath.Path(os.path.join(checkpoint_dir.rstrip("/"), "params"))
    print(f"Loading checkpoint from {params_dir}...", flush=True)
    handler = ocp.PyTreeCheckpointHandler()
    params = handler.restore(directory=params_dir, item=params)
    print("Checkpoint loaded.", flush=True)
    return params


def load_data(
    n_batches: int = 50,
    batch_size: int = 64,
) -> typing.List[np.ndarray]:
    """Load CIFAR-10 data batches."""
    print("Loading CIFAR-10...", flush=True)
    ds = datasets.load_dataset(
        "uoft-cs/cifar10",
        split="train",
        token=os.getenv("HF_TOKEN", None),
        revision="0b2714987fa478483af9968de7c934580d0bb9a2",
    )
    images = np.stack([np.array(item["img"]) for item in ds])
    # Normalize to [0, 1] then to [-1, 1]
    images = images.astype(np.float32) / 255.0
    images = images * 2.0 - 1.0

    rng = np.random.default_rng(42)
    batches = []
    for _ in range(n_batches):
        idx = rng.choice(len(images), size=batch_size, replace=False)
        batches.append(images[idx])
    print(f"Loaded {n_batches} batches of {batch_size}.", flush=True)
    return batches


# ==================================================================
# Experiment 1: Jacobian Variance Amplification
# ==================================================================
def experiment_1_variance_amplification(
    model: MeanFlowUNetModel,
    params: jaxtyping.PyTree,
    batches: typing.List[np.ndarray],
    t_values: typing.List[float],
    n_batches: int = 50,
) -> typing.Dict:
    """Measure per-sample loss variance with stochastic vs EMA tangent.

    For each t value, compute the MF loss with:
    (a) Stochastic tangent: v_cond = e - x0
    (b) Deterministic tangent: u(z, t, t) (self-evaluation)

    The variance ratio Var[stochastic] / Var[deterministic] should be >> 1,
    validating Theorem 1 (Jacobian variance amplification).
    """
    print("\n=== Experiment 1: Variance Amplification ===", flush=True)
    results = {}

    def compute_per_sample_loss(params, image, e, t_val, r_val, tangent_mode):
        """Compute per-sample squared residual."""
        batch_size = image.shape[0]
        t = jnp.full((batch_size,), t_val, dtype=jnp.float32)
        r = jnp.full((batch_size,), r_val, dtype=jnp.float32)
        z = (1 - t[:, None, None, None]) * image + t[:, None, None, None] * e
        v_cond = e - image

        def u_fn(z_t, r_in, t_in):
            timestamps = model._make_timestamps(t_in=t_in, r_in=r_in)
            return model._network.apply(
                variables={"params": params},
                inputs=z_t,
                timestamps=timestamps,
                edm_cond=None,
                deterministic=True,
            )

        if tangent_mode == "stochastic":
            # Use v_cond as tangent (vanilla MF)
            tangent_v = v_cond
        elif tangent_mode == "deterministic":
            # Use u(z, t, t) as tangent (VaMF-like)
            tangent_v = jax.lax.stop_gradient(u_fn(z, t, t))
        else:
            raise ValueError(f"Unknown tangent mode: {tangent_mode}")

        drdt = jnp.zeros_like(r)
        dtdt = jnp.ones_like(t)
        u, dudt = jax.jvp(u_fn, (z, r, t), (tangent_v, drdt, dtdt))
        u_target = v_cond - (t - r)[:, None, None, None] * dudt

        # Per-sample loss (sum over pixels)
        per_sample = jnp.sum(
            jnp.square(u - jax.lax.stop_gradient(u_target)),
            axis=(-1, -2, -3),
        )
        return per_sample

    jit_loss = jax.jit(
        compute_per_sample_loss, static_argnames=("tangent_mode",)
    )

    for t_val in t_values:
        stoch_losses, determ_losses = [], []
        for i in range(min(n_batches, len(batches))):
            image = jnp.array(batches[i])
            rng = jax.random.PRNGKey(i)
            e = jax.random.normal(rng, shape=image.shape)

            sl = jit_loss(params, image, e, t_val, 0.0, "stochastic")
            dl = jit_loss(params, image, e, t_val, 0.0, "deterministic")
            stoch_losses.append(np.array(sl))
            determ_losses.append(np.array(dl))

            if i % 10 == 0:
                print(f"  t={t_val:.1f}: batch {i}/{n_batches}", flush=True)

        stoch_all = np.concatenate(stoch_losses)
        determ_all = np.concatenate(determ_losses)

        var_stoch = float(np.var(stoch_all))
        var_determ = float(np.var(determ_all))
        ratio = var_stoch / max(var_determ, 1e-10)

        results[f"t={t_val:.1f}"] = {
            "stochastic_mean": float(np.mean(stoch_all)),
            "stochastic_var": var_stoch,
            "stochastic_std": float(np.std(stoch_all)),
            "deterministic_mean": float(np.mean(determ_all)),
            "deterministic_var": var_determ,
            "deterministic_std": float(np.std(determ_all)),
            "variance_ratio": ratio,
        }
        print(
            f"  t={t_val:.1f}: "
            f"Var[stoch]={var_stoch:.2f}, "
            f"Var[determ]={var_determ:.2f}, "
            f"ratio={ratio:.2f}",
            flush=True,
        )

    return results


# ==================================================================
# Experiment 2: Curvature Gap vs Interval Length
# ==================================================================
def experiment_2_curvature_gap(
    model: MeanFlowUNetModel,
    params: jaxtyping.PyTree,
    batches: typing.List[np.ndarray],
    t_values: typing.List[float],
    n_r_points: int = 20,
    n_batches: int = 10,
) -> typing.Dict:
    """Measure ||u(z, r, t) - v_cond||^2 as function of (t-r).

    For each fixed t, sweep r from 0 to t and plot the curvature gap.
    Theorem 3 predicts ||Delta||^2 ~ (t-r)^2.
    """
    print("\n=== Experiment 2: Curvature Gap ===", flush=True)
    results = {}

    def compute_curvature_gap(params, image, e, t_val, r_val):
        batch_size = image.shape[0]
        t = jnp.full((batch_size,), t_val, dtype=jnp.float32)
        r = jnp.full((batch_size,), r_val, dtype=jnp.float32)
        z = (1 - t[:, None, None, None]) * image + t[:, None, None, None] * e
        v_cond = e - image

        timestamps = model._make_timestamps(t_in=t, r_in=r)
        u = model._network.apply(
            variables={"params": params},
            inputs=z,
            timestamps=timestamps,
            edm_cond=None,
            deterministic=True,
        )

        # curvature gap: ||u(z,r,t) - v_cond||^2 per sample
        gap_sq = jnp.sum(jnp.square(u - v_cond), axis=(-1, -2, -3))
        return gap_sq

    jit_gap = jax.jit(compute_curvature_gap)

    for t_val in t_values:
        r_points = np.linspace(0.0, t_val, n_r_points + 1)[:-1]
        gaps_by_r = []

        for r_val in r_points:
            gap_samples = []
            for i in range(min(n_batches, len(batches))):
                image = jnp.array(batches[i])
                rng = jax.random.PRNGKey(i + 1000)
                e = jax.random.normal(rng, shape=image.shape)
                gap = jit_gap(params, image, e, t_val, float(r_val))
                gap_samples.append(np.array(gap))

            gap_all = np.concatenate(gap_samples)
            gaps_by_r.append(
                {
                    "r": float(r_val),
                    "t_minus_r": float(t_val - r_val),
                    "gap_sq_mean": float(np.mean(gap_all)),
                    "gap_sq_std": float(np.std(gap_all)),
                    "gap_mean": float(np.mean(np.sqrt(gap_all))),
                }
            )

        results[f"t={t_val:.1f}"] = gaps_by_r
        print(
            f"  t={t_val:.1f}: "
            f"gap at r=0: {gaps_by_r[0]['gap_sq_mean']:.2f}, "
            f"gap at r={r_points[-1]:.2f}: "
            f"{gaps_by_r[-1]['gap_sq_mean']:.2f}",
            flush=True,
        )

    return results


# ==================================================================
# Experiment 4: Jacobian Norm
# ==================================================================
def experiment_4_jacobian_norm(
    model: MeanFlowUNetModel,
    params: jaxtyping.PyTree,
    batches: typing.List[np.ndarray],
    t_values: typing.List[float],
    n_random_vectors: int = 5,
    n_batches: int = 10,
) -> typing.Dict:
    """Estimate Jacobian norm via Hutchinson estimator.

    ||d_z u||_F^2 ~ E[||d_z u * v||^2] where v ~ N(0, I).
    The Jacobian factor J = (t-r)*d_z u - I has norm that should
    grow with (t-r).
    """
    print("\n=== Experiment 4: Jacobian Norm ===", flush=True)
    results = {}

    def estimate_jac_norm(params, image, e, t_val, rand_vec):
        batch_size = image.shape[0]
        t = jnp.full((batch_size,), t_val, dtype=jnp.float32)
        r = jnp.zeros((batch_size,), dtype=jnp.float32)
        z = (1 - t[:, None, None, None]) * image + t[:, None, None, None] * e

        def u_z(z_in):
            timestamps = model._make_timestamps(t_in=t, r_in=r)
            return model._network.apply(
                variables={"params": params},
                inputs=z_in,
                timestamps=timestamps,
                edm_cond=None,
                deterministic=True,
            )

        # JVP to get d_z u * rand_vec
        _, jvp_out = jax.jvp(u_z, (z,), (rand_vec,))

        # ||d_z u * v||^2 per sample (Hutchinson estimate of ||d_z u||_F^2)
        jvp_norm_sq = jnp.sum(jnp.square(jvp_out), axis=(-1, -2, -3))

        # J = (t-r)*d_z u - I, so J*v = (t-r)*jvp_out - rand_vec
        j_times_v = t_val * jvp_out - rand_vec
        j_norm_sq = jnp.sum(jnp.square(j_times_v), axis=(-1, -2, -3))

        return jvp_norm_sq, j_norm_sq

    jit_jac = jax.jit(estimate_jac_norm)

    for t_val in t_values:
        jac_norms, j_norms = [], []
        for i in range(min(n_batches, len(batches))):
            image = jnp.array(batches[i])
            rng = jax.random.PRNGKey(i + 2000)
            e = jax.random.normal(rng, shape=image.shape)

            for rv_idx in range(n_random_vectors):
                rv_rng = jax.random.PRNGKey(i * 100 + rv_idx + 3000)
                rand_vec = jax.random.normal(rv_rng, shape=image.shape)
                jn, jfn = jit_jac(params, image, e, t_val, rand_vec)
                jac_norms.append(np.array(jn))
                j_norms.append(np.array(jfn))

        jac_all = np.concatenate(jac_norms)
        j_all = np.concatenate(j_norms)
        results[f"t={t_val:.1f}"] = {
            "dz_u_norm_sq_mean": float(np.mean(jac_all)),
            "dz_u_norm_sq_std": float(np.std(jac_all)),
            "J_norm_sq_mean": float(np.mean(j_all)),
            "J_norm_sq_std": float(np.std(j_all)),
            "J_norm_mean": float(np.mean(np.sqrt(j_all))),
        }
        print(
            f"  t={t_val:.1f}: "
            f"||d_z u||_F^2 ~ {np.mean(jac_all):.2f}, "
            f"||J||_F^2 ~ {np.mean(j_all):.2f}",
            flush=True,
        )

    return results


# ==================================================================
# Main
# ==================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to Orbax checkpoint (GCS or local).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/generative/vamf/results",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=50,
        help="Number of data batches for experiments.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="1,2,4",
        help="Comma-separated experiment numbers to run.",
    )
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("VaMF Diagnostic Measurements", flush=True)
    print(f"Checkpoint: {args.checkpoint_dir}", flush=True)
    print(f"Platform: {jax.default_backend()}", flush=True)
    print(f"Devices: {jax.devices()}", flush=True)
    print("=" * 60, flush=True)

    # Build model and load params
    model = build_model()
    params = load_params(model, args.checkpoint_dir)

    # Load data
    batches = load_data(n_batches=args.n_batches, batch_size=args.batch_size)

    experiments = [int(x) for x in args.experiments.split(",")]
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_results = {}

    if 1 in experiments:
        all_results[
            "exp1_variance_amplification"
        ] = experiment_1_variance_amplification(
            model,
            params,
            batches,
            t_values,
            n_batches=args.n_batches,
        )

    if 2 in experiments:
        all_results["exp2_curvature_gap"] = experiment_2_curvature_gap(
            model,
            params,
            batches,
            t_values,
            n_r_points=20,
            n_batches=min(10, args.n_batches),
        )

    if 4 in experiments:
        all_results["exp4_jacobian_norm"] = experiment_4_jacobian_norm(
            model,
            params,
            batches,
            t_values,
            n_random_vectors=5,
            n_batches=min(10, args.n_batches),
        )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "diagnostic_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}", flush=True)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)

    if "exp1_variance_amplification" in all_results:
        print("\nExp 1 — Variance Amplification Ratios:", flush=True)
        for k, v in all_results["exp1_variance_amplification"].items():
            print(
                f"  {k}: ratio = {v['variance_ratio']:.2f} "
                f"(stoch={v['stochastic_var']:.1f}, "
                f"determ={v['deterministic_var']:.1f})",
                flush=True,
            )

    if "exp4_jacobian_norm" in all_results:
        print("\nExp 4 — Jacobian Factor ||J||:", flush=True)
        for k, v in all_results["exp4_jacobian_norm"].items():
            print(
                f"  {k}: ||J||_F ~ {v['J_norm_mean']:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
