"""Entry point for running toy experiments with Mean Flows."""

import json
import math
import os
import time
import typing

from absl import app
from absl import flags
import chex
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random as jrnd
import jaxtyping
import optax
import typing_extensions

from src.core import model as _model
from src.core import train_state as _train_state
from src.utilities import logging as _logging

# ==============================================================================
# Flags
# ==============================================================================
flags.DEFINE_enum(
    name="dataset",
    default="checkerboard",
    enum_values=[
        "checkerboard",
        "eight_gaussians",
        "two_moons",
        "swiss_roll",
    ]
    + [f"gmm_{d}" for d in [2, 4, 8, 16]],
    help=(
        "Dataset to use, one of ['checkerboard', "
        "'eight_gaussians', 'two_moons', 'swiss_roll', "
        "'gmm_<d>']"
    ),
)
flags.DEFINE_enum(
    name="method",
    default="meanflow",
    enum_values=["meanflow", "vamf_l2", "vamf_tw", "ema_tw"],
    help=(
        "Method to run, one of ['meanflow', 'vamf_l2', " "'vamf_tw', 'ema_tw']"
    ),
)

# method hyperparameters
flags.DEFINE_integer(
    name="hidden_size",
    default=128,
    help="Dimensionality of the latent features in the MLPs.",
)
flags.DEFINE_integer(
    name="num_layers",
    default=3,
    help="Number of linear layers in the MLPs",
)
flags.DEFINE_float(
    name="ema_rate",
    default=0.999,
    help=(
        "Decay rate for the exponential moving average " "of the parameters."
    ),
)
flags.DEFINE_float(
    name="overlap_rate",
    default=0.25,
    help="Overlap ratio between `r` and `t` in MeanFlow.",
)
flags.DEFINE_float(
    name="max_gap",
    default=0.5,
    help="Maximum interval length (t - r).",
)
flags.DEFINE_enum(
    name="timestamp_sampler",
    default="uniform",
    enum_values=["uniform", "logit_normal"],
    help="Timestamp sampling strategy.",
)
flags.DEFINE_float(
    name="logit_normal_mean",
    default=0.0,
    help="Mean for logit-normal timestamp sampling.",
)
flags.DEFINE_float(
    name="logit_normal_stddev",
    default=1.0,
    help="Stddev for logit-normal timestamp sampling.",
)

# training hyperparameters
flags.DEFINE_integer(
    name="steps",
    default=20_000,
    help="Number of training steps.",
)
flags.DEFINE_integer(
    name="batch_size",
    default=256,
    help="Batch size for training and evaluation.",
)
flags.DEFINE_float(
    name="lr",
    default=0.001,
    help="Learning rate for the optimizer.",
)
flags.DEFINE_integer(
    name="seed",
    default=42,
    help="Random generator seed for reproducibility.",
)
flags.DEFINE_integer(
    name="log_every_n_steps",
    default=500,
    help="Log training metrics every N steps.",
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    required=True,
    help="Directory to save logs and checkpoints.",
)


################################################################################
# Datasets
def checkerboard(key: typing.Any, n: int) -> jax.Array:
    r"""Creates a four-by-four checker board with ``n`` samples.

    Args:
        key (Any): Random key generator.
        n (int): Number of samples to generate.

    Returns:
        An array of checkerboard samples in ``[-4, 4]`` x- and y-limits.
    """
    x_key, y_key, mask_key = jrnd.split(key, 3)
    x1 = jrnd.uniform(x_key, (n,)) * 4 - 2
    x2 = jrnd.uniform(y_key, (n,)) - (
        2.0 * jrnd.bernoulli(mask_key, 0.5, (n,)).astype(jnp.float32)
    )
    x2 = x2 + (jnp.floor(x1) % 2)

    return jnp.stack([x1, x2], axis=-1) * 2


def eight_gaussians(key: typing.Any, n: int) -> jax.Array:
    r"""Creates eight isotropic Gaussians on a circle.

    Args:
        key (Any): Random key generator.
        n (int): Number of samples to generate.

    Returns:
        An array of sample points from the eight isotropic Gaussians.
    """
    z_key, noise_key = jrnd.split(key, num=2)
    angles = jnp.linspace(0, 2 * jnp.pi, 9)[:-1]
    centers = 3.0 * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    idx = jrnd.randint(z_key, (n,), 0, 8)
    noise = jrnd.normal(noise_key, (n, 2)) * 0.25

    return centers[idx] + noise


def two_moons(key, n):
    r"""Two interleaving crescents.

    Args:
        key (Any): Random key generator.
        n (int): Number of samples to generate.

    Returns:
        An array of sample points from the two interleaving crescents.
    """
    k1, k2, k3 = jrnd.split(key, 3)
    n1, n2 = n // 2, n - n // 2
    t1 = jrnd.uniform(k1, (n1,)) * jnp.pi
    t2 = jrnd.uniform(k2, (n2,)) * jnp.pi
    noise = jrnd.normal(k3, (n, 2)) * 0.08
    upper = jnp.stack([jnp.cos(t1), jnp.sin(t1)], axis=-1)
    lower = jnp.stack([1 - jnp.cos(t2), -jnp.sin(t2) + 0.5], axis=-1)

    return (jnp.concatenate([upper, lower], axis=0) + noise) * 2


def swissroll(key, n):
    r"""2D Swiss roll.

    Args:
        key (Any): Random key generator.
        n (int): Number of samples to generate.

    Returns:
        An array of sample points from the swiss roll.
    """
    k1, k2 = jrnd.split(key)
    t = 1.5 * jnp.pi * (1 + 2 * jrnd.uniform(k1, (n,)))
    noise = jrnd.normal(k2, (n, 2)) * 0.08
    return jnp.stack([t * jnp.cos(t), t * jnp.sin(t)], -1) / 8 + noise


def gmm(key: typing.Any, n: int, d: int, k: int = 8) -> jax.Array:
    r"""Isotropic Gaussian mixture in ``d`` dimensions.

    Args:
        key (Any): Random key generator.
        n (int): Number of samples to generate.
        d (int): Dimensionality of each sample.
        k (int): Number of mixture components.

    Returns:
        An array of shape ``(n, d)``.
    """
    k1, k2, k3 = jrnd.split(key, 3)
    centers = jrnd.normal(k1, (k, d))
    centers = 3.0 * centers / jnp.linalg.norm(centers, axis=-1, keepdims=True)
    idx = jrnd.choice(k2, k, shape=(n,))
    return centers[idx] + 0.3 * jrnd.normal(k3, (n, d))


_DATASET_FN = {
    "checkerboard": checkerboard,
    "eight_gaussians": eight_gaussians,
    "two_moons": two_moons,
    "swiss_roll": swissroll,
}


def sample_data(key: jax.Array, dataset: str, n: int) -> jax.Array:
    """Sample a batch from the named dataset.

    Args:
        key: PRNG key.
        dataset: Dataset name.
        n: Number of samples.

    Returns:
        Array of shape ``(n, d)``.
    """
    if dataset in _DATASET_FN:
        return _DATASET_FN[dataset](key, n)
    if dataset.startswith("gmm_"):
        d = int(dataset.split("_")[1])
        return gmm(key, n, d)
    raise ValueError(f"Unknown dataset: {dataset}")


def data_dim(dataset: str) -> int:
    """Return the data dimensionality for a dataset name."""
    if dataset.startswith("gmm_"):
        return int(dataset.split("_")[1])
    return 2


################################################################################
# Timestamp Sampling
def sample_t_r(
    key: jax.Array,
    shape: typing.Tuple[int, ...],
    overlap_rate: float = 0.25,
    max_gap: float = 0.5,
    sampler: str = "uniform",
    logit_normal_mean: float = 0.0,
    logit_normal_stddev: float = 1.0,
) -> typing.Tuple[jax.Array, jax.Array]:
    """Sample ``(t, r)`` pairs for MeanFlow training.

    Args:
        key: PRNG key.
        shape: Batch shape.
        overlap_rate: Probability that ``r == t``.
        max_gap: Maximum ``(t - r)``.
        sampler: ``"uniform"`` or ``"logit_normal"``.
        logit_normal_mean: Mean for logit-normal sampling.
        logit_normal_stddev: Stddev for logit-normal sampling.

    Returns:
        Tuple ``(t, r)`` each of shape ``shape``.
    """
    t_key, r_key, mask_key = jrnd.split(key, 3)

    def logit_normal(
        key: jax.Array,
        shape: typing.Sequence[typing.Union[int, typing.Any]],
        dtype: typing.Any,
        mean: float,
        stddev: float,
    ) -> jax.Array:
        z = jax.random.normal(key=key, shape=shape, dtype=dtype)
        return jax.nn.sigmoid(mean + stddev * z)

    if sampler == "logit_normal":
        t = logit_normal(
            t_key,
            shape,
            jnp.float32,
            logit_normal_mean,
            logit_normal_stddev,
        )
        r = logit_normal(
            r_key,
            shape,
            jnp.float32,
            logit_normal_mean,
            logit_normal_stddev,
        )
        t = jnp.clip(t, 1e-4, 1.0 - 1e-4)
        r = jnp.clip(r, 1e-4, 1.0 - 1e-4)
    else:
        t = jrnd.uniform(
            t_key,
            shape,
            minval=1e-4,
            maxval=1.0 - 1e-4,
        )
        r = jrnd.uniform(
            r_key,
            shape,
            minval=1e-4,
            maxval=1.0 - 1e-4,
        )

    t, r = jnp.maximum(t, r), jnp.minimum(t, r)
    r = jnp.maximum(r, t - max_gap)
    mask = jrnd.uniform(mask_key, shape) < overlap_rate
    r = jnp.where(mask, t, r)
    return t, r


################################################################################
# Trace Weight (Proposition 3)
def trace_weight(
    u_fn: typing.Callable,
    z: jax.Array,
    r: jax.Array,
    t: jax.Array,
    key: jax.Array,
    n_probes: int = 1,
) -> jax.Array:
    r"""Per-sample weight ``1 / (1 + tr(JJ^T) / d)``.

    Uses a Hutchinson estimator with Rademacher probes.

    Args:
        u_fn: ``u_fn(z, r, t) -> (B, d)``.
        z: Noisy samples ``(B, d)``.
        r: Start timestamps ``(B,)``.
        t: End timestamps ``(B,)``.
        key: PRNG key.
        n_probes: Number of Hutchinson probes.

    Returns:
        Weights of shape ``(B,)``.
    """
    d = z.shape[-1]
    tr_jjt = jnp.zeros(z.shape[0])
    for i in range(n_probes):
        probe_key = jrnd.fold_in(key, i)
        v = jrnd.rademacher(probe_key, z.shape, dtype=z.dtype)
        _, jv = jax.jvp(lambda z_: u_fn(z_, r, t), (z,), (v,))
        tr_jjt = tr_jjt + jnp.sum(jv**2, axis=-1)
    tr_jjt = tr_jjt / n_probes
    return 1.0 / (1.0 + tr_jjt / d)


################################################################################
# Model
class MeanFlowMLPModule(nn.Module):
    r"""Multi-layer Perceptron with timestamp conditioning.

    Args:
        features (int): Dimensionality of the hidden features.
        num_layers (int): Number of layers.
        dtype (Any): Data type for the computations.
        param_dtype (Any): Data type for the parameters.
        precision (Any): Numerical precision for the computations.
    """

    features: int
    num_layers: int
    dtype: typing.Any
    param_dtype: typing.Any
    precision: typing.Any

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        r: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        r"""Forward pass the network.

        Args:
            inputs (jax.Array): Input noisy data with a shape of ``(*, d)``.
            r (jax.Array): Start timestamp of shape ``(*,)``.
            t (jax.Array): End timestamp of shape ``(*,)``.

        Returns:
            Predicted average displacement field of shape ``(*, d)``.
        """
        t_embed = self._sinusoidal_embedding(t)
        r_embed = self._sinusoidal_embedding(r)
        out = jnp.concatenate([inputs, t_embed, t_embed - r_embed], axis=-1)
        out = out.astype(self.dtype)

        for i in range(self.num_layers):
            out = nn.Dense(
                features=self.features,
                kernel_init=jax.nn.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="uniform",
                ),
                use_bias=True,
                bias_init=jax.nn.initializers.zeros,
                name=f"fc_{i}",
            )(out)
            out = nn.silu(out)

        out = nn.Dense(
            features=inputs.shape[-1],
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1e-10,
                mode="fan_avg",
                distribution="uniform",
            ),
            use_bias=True,
            bias_init=jax.nn.initializers.zeros,
            name="fc_out",
        )(out)

        return out

    @staticmethod
    def _sinusoidal_embedding(
        t: jax.Array,
        dim: int = 32,
    ) -> jax.Array:
        r"""Encode sinusoidal positional embedding."""
        half = dim // 2
        freqs = jnp.exp(
            -math.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / half
        )
        args = t[..., None] * freqs
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class MeanFlowMLPModel(_model.Model):
    def __init__(
        self,
        features: int,
        num_layers: int,
        method: str = "meanflow",
        max_gap: float = 0.5,
        overlap_rate: float = 0.25,
        timestamp_sampler: str = "uniform",
        logit_normal_mean: float = 0.0,
        logit_normal_stddev: float = 1.0,
        dtype: typing.Any = None,
        param_dtype: typing.Any = None,
        precision: typing.Any = None,
    ) -> None:
        self._dtype = dtype
        self._param_dtype = param_dtype
        self._method = method
        self._max_gap = max_gap
        self._overlap_rate = overlap_rate
        self._timestamp_sampler = timestamp_sampler
        self._logit_normal_mean = logit_normal_mean
        self._logit_normal_stddev = logit_normal_stddev
        self._network = MeanFlowMLPModule(
            features=features,
            num_layers=num_layers,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

    @property
    def network(self) -> MeanFlowMLPModule:
        return self._network

    @typing_extensions.override
    def init(
        self,
        *,
        batch: jax.Array,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Tuple[jaxtyping.PyTree, jaxtyping.PyTree]:
        del kwargs  # unused arguments

        assert isinstance(batch, jax.Array)
        r = jnp.ones_like(batch[..., 0])
        t = jnp.zeros_like(batch[..., 0])

        variables = self.network.init(
            inputs=batch,
            r=r,
            t=t,
            rngs=rngs,
        )
        assert isinstance(variables, dict)
        params = variables.pop("params")

        return params, variables

    @typing_extensions.override
    def forward(
        self,
        *,
        inputs: jax.Array,
        r: jax.Array,
        t: jax.Array,
        deterministic: bool = True,
        params: nn.FrozenDict,
        rngs: typing.Any,
        **kwargs,
    ) -> jax.Array:
        del deterministic, kwargs  # unused
        output = self.network.apply(
            variables=dict(params=params),
            inputs=inputs,
            r=r,
            t=t,
            rngs=rngs,
        )
        assert isinstance(output, jax.Array)
        chex.assert_equal_shape([inputs, output])

        return output

    @typing_extensions.override
    def training_step(
        self,
        *,
        batch: typing.Any,
        state: typing.Any,
        rngs: typing.Any,
        **kwargs,
    ) -> typing.Tuple[typing.Any, _model.StepOutputs]:
        x0 = batch

        def loss_fn(params):
            k_tr, k_e, k_tw = jrnd.split(rngs, 3)
            bsz = x0.shape[0]

            t, r = sample_t_r(
                k_tr,
                (bsz,),
                self._overlap_rate,
                self._max_gap,
                sampler=self._timestamp_sampler,
                logit_normal_mean=self._logit_normal_mean,
                logit_normal_stddev=self._logit_normal_stddev,
            )
            e = jrnd.normal(k_e, x0.shape, dtype=x0.dtype)
            z = (1.0 - t[..., None]) * x0 + t[..., None] * e
            v_cond = e - x0

            # JVP tangent: EMA prediction or stochastic
            if self._method in (
                "vamf_l2",
                "vamf_tw",
                "ema_tw",
            ):
                v_tang = jax.lax.stop_gradient(
                    self._network.apply(
                        {"params": state.ema_params},
                        z,
                        t,
                        t,
                    )
                )
            else:
                v_tang = v_cond

            def u_fn(z_in, r_in, t_in):
                return self._network.apply(
                    {"params": params},
                    z_in,
                    t_in,
                    r_in,
                )

            drdt = jnp.zeros_like(r)
            dtdt = jnp.ones_like(t)
            u, dudt = jax.jvp(
                u_fn,
                (z, r, t),
                (v_tang, drdt, dtdt),
            )

            gap = jnp.clip(t - r, a_min=0.0, a_max=1.0)
            v_pred = u + gap[..., None] * jax.lax.stop_gradient(dudt)
            v_target = jax.lax.stop_gradient(v_cond)
            per_sample = jnp.sum(
                jnp.square(v_pred - v_target),
                axis=-1,
            )

            # per-sample trace weight
            if self._method in ("vamf_tw", "ema_tw"):
                tw = trace_weight(
                    u_fn,
                    z,
                    r,
                    t,
                    k_tw,
                )
                weighted = per_sample * jax.lax.stop_gradient(tw)
            else:
                tw = jnp.ones(bsz)
                weighted = per_sample

            loss = jnp.mean(weighted)
            raw_loss = jnp.mean(per_sample)
            metrics = {
                "loss": loss,
                "raw_loss": raw_loss,
                "tw_mean": jnp.mean(tw),
            }
            return loss, metrics

        grads, metrics = jax.grad(
            loss_fn,
            has_aux=True,
        )(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, _model.StepOutputs(
            scalars=metrics,
        )

    @typing_extensions.override
    def evaluation_step(
        self,
        *,
        batch: typing.Any,
        params: nn.FrozenDict,
        rngs: typing.Any,
        **kwargs,
    ) -> _model.StepOutputs:
        r"""One-step generation: ``x0 = z1 - u(z1, t=1, r=0)``."""
        n = batch.shape[0]
        d = batch.shape[-1]
        z1 = jrnd.normal(rngs, (n, d))
        t = jnp.ones(n)
        r = jnp.zeros(n)
        u = self._network.apply(
            {"params": params},
            z1,
            t,
            r,
        )
        return _model.StepOutputs(output=z1 - u)


################################################################################
# Main
def main(argv: typing.List[str]) -> int:
    del argv  # unused arguments

    FLAGS = flags.FLAGS
    key = jrnd.PRNGKey(FLAGS.seed)
    d = data_dim(FLAGS.dataset)

    # ---- build model ----
    _logging.rank_zero_info("Building model...")
    key, init_key = jrnd.split(key, num=2)
    model = MeanFlowMLPModel(
        features=FLAGS.hidden_size,
        num_layers=FLAGS.num_layers,
        method=FLAGS.method,
        max_gap=FLAGS.max_gap,
        overlap_rate=FLAGS.overlap_rate,
        timestamp_sampler=FLAGS.timestamp_sampler,
        logit_normal_mean=FLAGS.logit_normal_mean,
        logit_normal_stddev=FLAGS.logit_normal_stddev,
    )
    params, _ = model.init(
        batch=jnp.zeros((1, d)),
        rngs=init_key,
    )
    _logging.rank_zero_info("Building model... DONE!")

    # ---- build train state ----
    _logging.rank_zero_info("Building train state...")
    tx = optax.adam(learning_rate=FLAGS.lr)
    state = _train_state.TrainState.create(
        params=params,
        tx=tx,
        ema_rate=FLAGS.ema_rate,
    )
    state = jax.block_until_ready(state)
    _logging.rank_zero_info("Building train state... DONE!")

    # ---- jit-compiled training step ----
    def _train_step(state, key):
        k_data, k_loss = jrnd.split(key)
        x0 = sample_data(
            k_data,
            FLAGS.dataset,
            FLAGS.batch_size,
        )
        return model.training_step(
            batch=x0,
            state=state,
            rngs=k_loss,
        )

    train_step = jax.jit(_train_step)

    # ---- training loop ----
    _logging.rank_zero_info(
        "Training %s on %s for %d steps...",
        FLAGS.method,
        FLAGS.dataset,
        FLAGS.steps,
    )
    history = []
    t0 = time.time()

    for step in range(FLAGS.steps):
        key, step_key = jrnd.split(key)
        state, step_out = train_step(state, step_key)

        if step % FLAGS.log_every_n_steps == 0 or step == FLAGS.steps - 1:
            m = {k: float(v) for k, v in step_out.scalars.items()}
            m["step"] = step
            history.append(m)
            elapsed = time.time() - t0
            _logging.rank_zero_info(
                "[%6d/%d] raw_loss=%.4f  loss=%.4f  " "tw=%.4f  (%.1fs)",
                step,
                FLAGS.steps,
                m["raw_loss"],
                m["loss"],
                m["tw_mean"],
                elapsed,
            )

    elapsed = time.time() - t0
    _logging.rank_zero_info("Training finished in %.1fs.", elapsed)

    # ---- save results ----
    os.makedirs(FLAGS.work_dir, exist_ok=True)
    fname = f"{FLAGS.dataset}_{FLAGS.method}_{FLAGS.seed}.json"
    out_path = os.path.join(FLAGS.work_dir, fname)
    final = history[-1] if history else {}
    final["elapsed_s"] = elapsed
    final["method"] = FLAGS.method
    final["dataset"] = FLAGS.dataset
    with open(out_path, "w") as f:
        json.dump(
            {
                "args": {
                    "dataset": FLAGS.dataset,
                    "method": FLAGS.method,
                    "steps": FLAGS.steps,
                    "batch_size": FLAGS.batch_size,
                    "lr": FLAGS.lr,
                    "hidden_size": FLAGS.hidden_size,
                    "num_layers": FLAGS.num_layers,
                    "ema_rate": FLAGS.ema_rate,
                    "max_gap": FLAGS.max_gap,
                    "overlap_rate": FLAGS.overlap_rate,
                    "timestamp_sampler": FLAGS.timestamp_sampler,
                    "logit_normal_mean": FLAGS.logit_normal_mean,
                    "logit_normal_stddev": FLAGS.logit_normal_stddev,
                    "seed": FLAGS.seed,
                },
                "history": history,
                "final": final,
            },
            f,
            indent=2,
        )
    _logging.rank_zero_info("Saved results to %s", out_path)

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
