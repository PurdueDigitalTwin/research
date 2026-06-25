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
from jax import flatten_util
from jax import numpy as jnp
from jax import random as jrnd
import jaxtyping
import numpy as np
import optax
import typing_extensions

from src.core import model as _model
from src.core import train_state as _train_state
from src.projects.generative.vamf.model import beta_schedule as _beta_schedule
from src.projects.generative.vamf.model import trace
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
        "two_spirals",
        "pinwheel",
    ]
    + [f"gmm_{d}" for d in [2, 4, 8, 16]]
    + [f"dgmm_{d}" for d in [2, 4, 8, 16, 32, 64]],
    help=(
        "Dataset to use, one of ['checkerboard', "
        "'eight_gaussians', 'two_moons', 'swiss_roll', "
        "'two_spirals', 'pinwheel', 'gmm_<d>']"
    ),
)
flags.DEFINE_enum(
    name="method",
    default="meanflow",
    enum_values=[
        "meanflow",
        "vamf_l2",
        "vamf_tw",
        "vamf_anneal",
        "vamf_tmix",
    ],
    help=(
        "Method to run. 'vamf_anneal' mixes v_cond and u_bar(x_t,t,t) in "
        "the regression target (Theorem 2). 'vamf_tmix' mixes them in the "
        "JVP tangent (Theorem 3): tangent = (1-tangent_beta)*v_cond + "
        "tangent_beta*u_bar. tangent_beta=0 recovers MeanFlow; "
        "tangent_beta=1 recovers VaMF-L2."
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
flags.DEFINE_integer(
    name="tw_n_probes",
    default=1,
    help="Number of Hutchinson probes for trace weight estimation.",
)
flags.DEFINE_float(
    name="target_alpha",
    default=0.0,
    help=(
        "Mixing weight alpha in [0, 1] for the annealed regression target "
        "v(alpha) = (1-alpha)*v_cond + alpha*u_bar(x_t,t,t). Only used by "
        "method=vamf_anneal. alpha=0 recovers VaMF-L2; alpha=1 is full "
        "EMA-target."
    ),
)
flags.DEFINE_float(
    name="tangent_beta",
    default=1.0,
    help=(
        "Mixing weight beta in [0, 1] for the JVP tangent (Theorem 3): "
        "tangent = (1-beta)*v_cond + beta*u_bar(x_t,t,t). "
        "Only used by method=vamf_tmix. beta=0 recovers MeanFlow; "
        "beta=1 recovers VaMF-L2 (full EMA tangent)."
    ),
)
flags.DEFINE_enum(
    name="beta_anneal_shape",
    default="constant",
    enum_values=["constant", "linear", "cosine", "step"],
    help=(
        "vamf_tmix tangent-beta schedule; "
        "'constant' == static --tangent_beta."
    ),
)
flags.DEFINE_float(
    name="beta_start",
    default=1.0,
    help="beta at step 0.",
)
flags.DEFINE_float(
    name="beta_end",
    default=0.0,
    help="beta after anneal window.",
)
flags.DEFINE_float(
    name="beta_anneal_s0",
    default=0.0,
    help="anneal start / steps.",
)
flags.DEFINE_float(
    name="beta_anneal_s1",
    default=0.6,
    help="anneal end / steps.",
)
flags.DEFINE_boolean(
    name="exact_trace",
    default=False,
    help="Compute exact Jacobian trace (for low-d toy experiments).",
)
flags.DEFINE_enum(
    name="tw_sigma",
    default="none",
    enum_values=[
        "none",  # sigma_t = 1                   (constant)
        "t",  # sigma_t = t                   (variance ~ t^2)
        "t_squared",  # sigma_t = t^2                 (variance ~ t^4)
        "ushape",  # sigma_t = sqrt(t * (1-t))     (variance peaks at t=0.5)
        "blue_gauss",  # sigma_t^2 = 1/((1-t)^2 + t^2) (Gaussian-data BLUE)
        "learned",  # sigma_t = NN_phi(t)           (small MLP)
    ],
    help=(
        "Sigma schedule for trace weight. The schedule sets the per-sample "
        "standard-deviation scale sigma_t; the trace weight uses sigma_t^2 "
        "in the denominator (BLUE-scalar form)."
    ),
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
flags.DEFINE_integer(
    name="measure_grad_var_every",
    default=0,
    help=(
        "If > 0, measure per-step gradient noise ratio every N steps. "
        "0 disables the diagnostic."
    ),
)
flags.DEFINE_integer(
    name="measure_grad_var_n_batches",
    default=8,
    help=(
        "Number of independent mini-batches used to estimate gradient "
        "covariance and mean per measurement point."
    ),
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


def two_spirals(key, n):
    r"""Two interleaved Archimedean spirals.

    Args:
        key (Any): Random key generator.
        n (int): Number of samples to generate.

    Returns:
        An array of sample points from two spirals that interleave.
    """
    k_t, k_branch, k_noise = jrnd.split(key, 3)
    n_per = n // 2
    t = jnp.sqrt(jrnd.uniform(k_t, (n,))) * 540.0 * jnp.pi / 180.0
    sign = jnp.where(jnp.arange(n) < n_per, 1.0, -1.0)
    x = sign * t * jnp.cos(t) / 5.0
    y = sign * t * jnp.sin(t) / 5.0
    pts = jnp.stack([x, y], axis=-1)
    noise = jrnd.normal(k_noise, (n, 2)) * 0.08
    del k_branch  # unused
    return pts + noise


def pinwheel(key, n, num_arms: int = 5):
    r"""Pinwheel mixture: ``num_arms`` curved Gaussian arms.

    Each component is a Gaussian rotated and sheared so it traces a
    spiral arm. The arms intersect near the origin, creating heavy
    mode-mixing and high local curvature.

    Args:
        key (Any): Random key generator.
        n (int): Number of samples to generate.
        num_arms (int): Number of arms. Defaults to 5.

    Returns:
        An array of sample points.
    """
    k_idx, k_xy = jrnd.split(key, 2)
    rate = 0.25
    rads = jnp.linspace(0.0, 2.0 * jnp.pi, num_arms + 1)[:-1]
    radial_std = 0.3
    tangential_std = 0.1

    idx = jrnd.choice(k_idx, num_arms, shape=(n,))
    base = jrnd.normal(k_xy, (n, 2)) * jnp.array([radial_std, tangential_std])
    base = base + jnp.array([1.0, 0.0])
    angles = rads[idx] + rate * jnp.exp(base[:, 0])
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    rot = jnp.stack(
        [jnp.stack([cos, -sin], axis=-1), jnp.stack([sin, cos], axis=-1)],
        axis=-2,
    )
    return jnp.einsum("nij,nj->ni", rot, base) * 1.5


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


def dense_gmm(
    key: typing.Any,
    n: int,
    d: int,
    k: int = 64,
) -> jax.Array:
    r"""Dense Gaussian mixture with overlapping components.

    Uses more components with smaller separation to create significant
    path overlap, making variance amplification a dominant effect.

    Args:
        key: Random key generator.
        n: Number of samples to generate.
        d: Dimensionality of each sample.
        k: Number of mixture components.

    Returns:
        An array of shape ``(n, d)``.
    """
    k1, k2, k3 = jrnd.split(key, 3)
    centers = jrnd.normal(k1, (k, d))
    centers = 1.5 * centers / jnp.linalg.norm(centers, axis=-1, keepdims=True)
    idx = jrnd.choice(k2, k, shape=(n,))
    return centers[idx] + 0.5 * jrnd.normal(k3, (n, d))


_DATASET_FN = {
    "checkerboard": checkerboard,
    "eight_gaussians": eight_gaussians,
    "two_moons": two_moons,
    "swiss_roll": swissroll,
    "two_spirals": two_spirals,
    "pinwheel": pinwheel,
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
    if dataset.startswith("dgmm_"):
        d = int(dataset.split("_")[1])
        return dense_gmm(key, n, d)
    if dataset.startswith("gmm_"):
        d = int(dataset.split("_")[1])
        return gmm(key, n, d)
    raise ValueError(f"Unknown dataset: {dataset}")


def data_dim(dataset: str) -> int:
    """Return the data dimensionality for a dataset name."""
    if dataset.startswith("dgmm_"):
        return int(dataset.split("_")[1])
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
    sigma_t: typing.Optional[jax.Array] = None,
    n_probes: int = 1,
    exact: bool = False,
) -> jax.Array:
    r"""Returns per-sample weight :math:`1 / (1 + sigam_t * tr(B B^T) / d)`.

    Args:
        u_fn: ``u_fn(z, r, t) -> (B, d)``.
        z: Noisy samples ``(B, d)``.
        r: Start timestamps ``(B,)``.
        t: End timestamps ``(B,)``.
        key: PRNG key.
        sigma_t: Per-sample scaling ``(B,)``. Defaults to ones.
        n_probes: Number of Hutchinson probes (ignored if exact=True).
        exact: If True, compute exact Jacobian (good for small d).

    Returns:
        Weights of shape ``(B,)``.
    """
    d = z.shape[-1]
    if exact:
        tr_bbt = trace.exact_trace(u_fn, z, r, t)
    else:
        tr_bbt = trace.hutchinson_trace(key, u_fn, z, r, t, n_probes)
    if sigma_t is None:
        sigma_t = jnp.ones_like(t)
    # Sigma_t is the conditional-velocity *standard deviation* scale; the
    # BLUE-scalar trace weight has sigma_t^2 in the denominator.
    return 1.0 / (1.0 + jnp.square(sigma_t) * tr_bbt / d)


################################################################################
# Quality Metrics
def sliced_wasserstein(
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    n_projections: int = 500,
    p: float = 1.0,
) -> jax.Array:
    r"""Sliced Wasserstein-``p`` distance between two empirical distributions.

    Estimates :math:`SW_p(P, Q) = \left(\mathbb{E}_\theta
    [W_p(\theta_\# P, \theta_\# Q)^p]\right)^{1/p}` via Monte Carlo over
    uniformly random unit directions :math:`\theta`. For empirical measures
    of equal size, the inner 1-D Wasserstein distance reduces to comparing
    sorted samples.

    Args:
        x: Samples from P, shape ``(n, d)``.
        y: Samples from Q, shape ``(n, d)``.
        key: PRNG key for random projection directions.
        n_projections: Number of random 1-D projections.
        p: Order of the Wasserstein distance. Defaults to 1.

    Returns:
        Scalar :math:`SW_p` estimate.
    """
    d = x.shape[-1]
    dirs = jrnd.normal(key, (n_projections, d))
    dirs = dirs / jnp.linalg.norm(
        dirs,
        axis=-1,
        keepdims=True,
    )
    x_proj = jnp.sort(x @ dirs.T, axis=0)
    y_proj = jnp.sort(y @ dirs.T, axis=0)
    return jnp.mean(jnp.abs(x_proj - y_proj) ** p) ** (1.0 / p)


################################################################################
# Learnable Sigma
class SigmaModule(nn.Module):
    r"""Learnable positive scalar ``σ(t)`` for trace weight scaling."""

    @nn.compact
    def __call__(self, t: jax.Array) -> jax.Array:
        x = t[..., None]
        x = nn.Dense(32, name="fc_0")(x)
        x = nn.silu(x)
        x = nn.Dense(1, name="fc_out")(x)
        return nn.softplus(x[..., 0])


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
        tw_n_probes: int = 1,
        exact_trace: bool = False,
        tw_sigma: str = "none",
        target_alpha: float = 0.0,
        tangent_beta: float = 1.0,
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
        self._tw_n_probes = tw_n_probes
        self._exact_trace = exact_trace
        self._tw_sigma = tw_sigma
        self._target_alpha = float(target_alpha)
        self._tangent_beta = float(tangent_beta)
        self._network = MeanFlowMLPModule(
            features=features,
            num_layers=num_layers,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self._sigma_net = SigmaModule() if tw_sigma == "learned" else None

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

        if self._sigma_net is not None:
            sigma_vars = self._sigma_net.init(rngs, t)
            params = {"velocity": params, "sigma": sigma_vars["params"]}
        else:
            params = {"velocity": params}

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
        vel_params = params.get("velocity", params)
        output = self.network.apply(
            variables=dict(params=vel_params),
            inputs=inputs,
            r=r,
            t=t,
            rngs=rngs,
        )
        assert isinstance(output, jax.Array)
        chex.assert_equal_shape([inputs, output])

        return output

    def _loss_fn_and_aux(self, batch, state, rngs, tangent_beta=None):
        """Build the loss closure used by training and gradient probes.

        Returns (loss_fn, aux) where loss_fn(params) -> (loss, metrics).
        Both ``training_step`` and ``compute_gradient`` use this so that
        the gradient-variance diagnostic measures the *same* loss surface
        that training optimizes.
        """
        x0 = batch

        def loss_fn(params):
            k_tr, k_e, k_tw = jrnd.split(rngs, 3)
            bsz = x0.shape[0]
            vel_params = params.get("velocity", params)
            ema_vel = state.ema_params.get("velocity", state.ema_params)

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

            # JVP tangent: pure stochastic, pure EMA, or beta-mixed.
            if self._method in (
                "vamf_l2",
                "vamf_tw",
                "vamf_anneal",
            ):
                v_tang = jax.lax.stop_gradient(
                    self._network.apply(
                        {"params": ema_vel},
                        z,
                        t,
                        t,
                    )
                )
            elif self._method == "vamf_tmix":
                # NOTE: v_{tangent} = (1-beta) * v_cond + beta * u_bar(x_t,t,t)
                ema_v = jax.lax.stop_gradient(
                    self._network.apply(
                        {"params": ema_vel},
                        z,
                        t,
                        t,
                    )
                )
                assert isinstance(ema_v, jax.Array)
                b = (
                    self._tangent_beta
                    if tangent_beta is None
                    else tangent_beta
                )
                v_tang = (1.0 - b) * v_cond + b * ema_v
            else:
                v_tang = v_cond
            assert isinstance(v_tang, jax.Array)

            def u_fn(z_in, r_in, t_in):
                return self._network.apply(
                    {"params": vel_params},
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
            # Annealed regression target: (1-alpha)*v_cond + alpha*u_bar(x_t,t,t).
            # alpha=0 recovers the original v_cond target (VaMF-L2 / VaMF-TW
            # behavior); alpha=1 is full EMA-target. v_tang already holds
            # u_bar(x_t,t,t) under stop-gradient for the EMA methods.
            if self._method == "vamf_anneal" and self._target_alpha > 0.0:
                a = self._target_alpha
                v_target = jax.lax.stop_gradient(
                    (1.0 - a) * v_cond + a * v_tang
                )
            else:
                v_target = jax.lax.stop_gradient(v_cond)
            per_sample = jnp.sum(
                jnp.square(v_pred - v_target),
                axis=-1,
            )

            # per-sample trace weight
            if self._method == "vamf_tw":
                # Compute sigma_t schedule (sigma_t is the std-dev scale;
                # trace weight uses sigma_t^2 in the denominator).
                if self._tw_sigma == "t":
                    sigma_t = t
                elif self._tw_sigma == "t_squared":
                    sigma_t = t**2
                elif self._tw_sigma == "ushape":
                    # sqrt(t*(1-t)): peaks at t=0.5, zero at endpoints.
                    sigma_t = jnp.sqrt(jnp.clip(t * (1.0 - t), a_min=1e-8))
                elif self._tw_sigma == "blue_gauss":
                    # Closed-form sigma_t^2 = 1/((1-t)^2 + t^2) under the
                    # Gaussian-data approximation x_0 ~ N(0, I); we store
                    # the std as sqrt of that.
                    sigma_t = 1.0 / jnp.sqrt(
                        jnp.square(1.0 - t) + jnp.square(t)
                    )
                elif self._tw_sigma == "learned" and isinstance(
                    self._sigma_net, SigmaModule
                ):
                    sigma_t = self._sigma_net.apply(
                        {"params": params["sigma"]},
                        t,
                    )
                else:
                    sigma_t = jnp.ones_like(t)
                assert isinstance(sigma_t, jax.Array)

                tw = trace_weight(
                    u_fn,
                    z,
                    r,
                    t,
                    k_tw,
                    sigma_t=sigma_t,
                    n_probes=self._tw_n_probes,
                    exact=self._exact_trace,
                )
                weighted = per_sample * jax.lax.stop_gradient(tw)
            else:
                sigma_t = jnp.ones(bsz)
                tw = jnp.ones(bsz)
                weighted = per_sample

            loss = jnp.mean(weighted)
            raw_loss = jnp.mean(per_sample)
            metrics = {
                "loss": loss,
                "raw_loss": raw_loss,
                "tw_mean": jnp.mean(tw),
                "sigma_mean": jnp.mean(sigma_t),
            }
            return loss, metrics

        return loss_fn, None

    @typing_extensions.override
    def training_step(
        self,
        *,
        batch: typing.Any,
        state: typing.Any,
        rngs: typing.Any,
        tangent_beta=None,
        **kwargs,
    ) -> typing.Tuple[typing.Any, _model.StepOutputs]:
        loss_fn, _ = self._loss_fn_and_aux(
            batch, state, rngs, tangent_beta=tangent_beta
        )
        grads, metrics = jax.grad(
            loss_fn,
            has_aux=True,
        )(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, _model.StepOutputs(
            scalars=metrics,
        )

    def compute_gradient(
        self,
        *,
        batch: typing.Any,
        state: typing.Any,
        rngs: typing.Any,
        tangent_beta=None,
    ) -> typing.Any:
        """Returns the raw parameter gradient for a single (batch, rngs) pair.

        Reuses the exact loss surface of
        ``training_step`` but does not apply the gradient. Used by the
        gradient-variance diagnostic (Theorem 3 validation).
        """
        loss_fn, _ = self._loss_fn_and_aux(
            batch, state, rngs, tangent_beta=tangent_beta
        )
        grads, _ = jax.grad(loss_fn, has_aux=True)(state.params)
        return grads

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
        vel_params = params.get("velocity", params)
        n = batch.shape[0]
        d = batch.shape[-1]
        z1 = jrnd.normal(rngs, (n, d))
        t = jnp.ones(n)
        r = jnp.zeros(n)
        u = self._network.apply(
            {"params": vel_params},
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
        tw_n_probes=FLAGS.tw_n_probes,
        exact_trace=FLAGS.exact_trace,
        tw_sigma=FLAGS.tw_sigma,
        target_alpha=FLAGS.target_alpha,
        tangent_beta=FLAGS.tangent_beta,
    )
    params, _ = model.init(
        batch=jnp.zeros((1, d)),
        rngs=init_key,
    )
    _logging.rank_zero_info("Building model... DONE!")

    # ---- build train state ----
    _logging.rank_zero_info("Building train state...")
    lr_schedule = optax.cosine_decay_schedule(
        init_value=FLAGS.lr,
        decay_steps=FLAGS.steps,
        alpha=0.01,
    )
    tx = optax.adam(learning_rate=lr_schedule)
    state = _train_state.TrainState.create(
        params=params,
        tx=tx,
        ema_rate=FLAGS.ema_rate,
    )
    state = jax.block_until_ready(state)
    _logging.rank_zero_info("Building train state... DONE!")

    # ---- jit-compiled training step ----
    def _train_step(state, key, beta_val):
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
            tangent_beta=beta_val,
        )

    train_step = jax.jit(_train_step)

    # ---- jit-compiled gradient-variance probe ----
    # Calls model.compute_gradient with an independent (data, loss) rng
    # split, returning the flattened parameter gradient. We invoke this
    # K times with K independent keys per measurement step to estimate
    # the per-step gradient mean and variance (Theorem 3 diagnostic).
    def _grad_probe(state, key, beta_val):
        k_data, k_loss = jrnd.split(key)
        x0 = sample_data(k_data, FLAGS.dataset, FLAGS.batch_size)
        grads = model.compute_gradient(
            batch=x0, state=state, rngs=k_loss, tangent_beta=beta_val
        )
        flat, _ = flatten_util.ravel_pytree(grads)
        return flat

    grad_probe = jax.jit(_grad_probe)

    # ---- jit-compiled eval step (SWD) ----
    n_eval = 4096

    def _eval_step(state, key):
        k_ref, k_gen, k_swd = jrnd.split(key, 3)
        ref = sample_data(
            k_ref,
            FLAGS.dataset,
            n_eval,
        )
        gen_out = model.evaluation_step(
            batch=ref,
            params=state.ema_params,
            rngs=k_gen,
        )
        assert gen_out.output is not None
        # Same projection key for both p so SW_1 / SW_2 are paired estimators
        # over an identical set of slicing directions (variance-reduced).
        swd1 = sliced_wasserstein(gen_out.output, ref, k_swd, p=1.0)
        swd2 = sliced_wasserstein(gen_out.output, ref, k_swd, p=2.0)
        return swd1, swd2

    eval_step = jax.jit(_eval_step)

    # ---- training loop ----
    _logging.rank_zero_info(
        "Training %s on %s for %d steps...",
        FLAGS.method,
        FLAGS.dataset,
        FLAGS.steps,
    )
    history = []
    t0 = time.time()

    grad_var_history = []  # diagnostic: per-step gradient noise ratio

    for step in range(FLAGS.steps):
        key, step_key = jrnd.split(key)
        if FLAGS.beta_anneal_shape == "constant":
            beta_val = jnp.asarray(
                FLAGS.tangent_beta, dtype=jnp.float32
            )
        else:
            beta_val = jnp.asarray(
                _beta_schedule.beta_at_step(
                    step,
                    FLAGS.steps,
                    shape=FLAGS.beta_anneal_shape,
                    beta_start=FLAGS.beta_start,
                    beta_end=FLAGS.beta_end,
                    s0=FLAGS.beta_anneal_s0,
                    s1=FLAGS.beta_anneal_s1,
                ),
                dtype=jnp.float32,
            )
        state, step_out = train_step(state, step_key, beta_val)

        # Gradient-variance probe (Theorem 3 diagnostic).
        if (
            FLAGS.measure_grad_var_every > 0
            and step % FLAGS.measure_grad_var_every == 0
        ):
            K = FLAGS.measure_grad_var_n_batches
            key, *probe_keys = jrnd.split(key, K + 1)
            grads_flat = jnp.stack(
                [grad_probe(state, kk, beta_val) for kk in probe_keys],
                axis=0,
            )  # (K, P)
            mean_grad = jnp.mean(grads_flat, axis=0)
            mean_norm_sq = float(jnp.sum(jnp.square(mean_grad)))
            # Sample variance trace = (1/(K-1)) sum_k ||g_k - mean||^2
            tr_cov = float(
                jnp.sum(jnp.square(grads_flat - mean_grad[None, :])) / (K - 1)
            )
            nr = tr_cov / max(mean_norm_sq, 1e-30)
            grad_var_history.append(
                {
                    "step": step,
                    "tr_cov": tr_cov,
                    "mean_norm_sq": mean_norm_sq,
                    "nr": nr,
                }
            )

        if step % FLAGS.log_every_n_steps == 0 or step == FLAGS.steps - 1:
            key, eval_key = jrnd.split(key)
            swd1, swd2 = eval_step(state, eval_key)
            m = {k: float(v) for k, v in step_out.scalars.items()}
            m["step"] = step
            m["swd1"] = float(swd1)
            m["swd2"] = float(swd2)
            history.append(m)
            elapsed = time.time() - t0
            _logging.rank_zero_info(
                "[%6d/%d] loss=%.4f  swd1=%.4f  swd2=%.4f  (%.1fs)",
                step,
                FLAGS.steps,
                m["loss"],
                m["swd1"],
                m["swd2"],
                elapsed,
            )

    elapsed = time.time() - t0
    _logging.rank_zero_info("Training finished in %.1fs.", elapsed)

    # ---- save results ----
    os.makedirs(FLAGS.work_dir, exist_ok=True)
    fname = (
        f"{FLAGS.dataset}_{FLAGS.method}"
        f"_{FLAGS.beta_anneal_shape}_s{FLAGS.beta_anneal_s1}"
        f"_b{FLAGS.tangent_beta}_{FLAGS.seed}.json"
    )
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
                    "tw_n_probes": FLAGS.tw_n_probes,
                    "exact_trace": FLAGS.exact_trace,
                    "tw_sigma": FLAGS.tw_sigma,
                    "target_alpha": FLAGS.target_alpha,
                    "tangent_beta": FLAGS.tangent_beta,
                    "beta_anneal_shape": FLAGS.beta_anneal_shape,
                    "beta_start": FLAGS.beta_start,
                    "beta_end": FLAGS.beta_end,
                    "beta_anneal_s0": FLAGS.beta_anneal_s0,
                    "beta_anneal_s1": FLAGS.beta_anneal_s1,
                    "seed": FLAGS.seed,
                },
                "history": history,
                "grad_var_history": grad_var_history,
                "final": final,
            },
            f,
            indent=2,
        )
    _logging.rank_zero_info("Saved results to %s", out_path)

    # ---- generate and save samples ----
    n_gen = 4096
    key, gen_key, ref_key = jrnd.split(key, 3)
    ref_data = sample_data(ref_key, FLAGS.dataset, n_gen)
    gen_out = model.evaluation_step(
        batch=ref_data,
        params=state.ema_params,
        rngs=gen_key,
    )
    npz_path = os.path.join(
        FLAGS.work_dir,
        f"{FLAGS.dataset}_{FLAGS.method}"
        f"_{FLAGS.beta_anneal_shape}_s{FLAGS.beta_anneal_s1}"
        f"_b{FLAGS.tangent_beta}_{FLAGS.seed}.npz",
    )
    np.savez(
        npz_path,
        generated=np.asarray(gen_out.output),
        reference=np.asarray(ref_data),
    )
    _logging.rank_zero_info("Saved samples to %s", npz_path)

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
