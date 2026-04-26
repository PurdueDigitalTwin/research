"""Entry point for running toy experiments with Mean Flows."""

import math
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
    enum_values=["checkerboard", "eight_gaussians", "two_moons", "swiss_roll"]
    + [f"gmm_{d}" for d in [2, 4, 8, 16]],
    help="Dataset to use, one of ['checkerboard', 'eight_gaussians', 'two_moons', 'swiss_roll', 'gmm_<d>']",
)
flags.DEFINE_enum(
    name="method",
    default="meanflow",
    enum_values=["meanflow", "vamf_l2", "vamf_tw", "ema_tw"],
    help="Method to run, one of ['meanflow', 'vamf_l2', 'vamf_tw', 'ema_tw']",
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
    help="Decay rate for the exponential moving average of the parameters.",
)
flags.DEFINE_float(
    name="overlap_rate",
    default=0.25,
    help="Overlap ratio between `r` and `t` in MeanFlow.",
)

# training hyperparameters
flags.DEFINE_integer(
    name="steps",
    default=None,
    help="Number of training steps. If not specified, run until convergence.",
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
    x_key, y_key, mask_key = jax.random.split(key, 3)
    x1 = jax.random.uniform(x_key, (n,)) * 4 - 2
    x2 = jax.random.uniform(y_key, (n,)) - (
        2.0 * jax.random.bernoulli(mask_key, 0.5, (n,)).astype(jnp.float32)
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
    z_key, noise_key = jax.random.split(key, num=2)
    angles = jnp.linspace(0, 2 * jnp.pi, 9)[:-1]
    centers = 3.0 * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    idx = jax.random.randint(z_key, (n,), 0, 8)
    noise = jax.random.normal(noise_key, (n, 2)) * 0.25

    return centers[idx] + noise


def two_moons(key, n):
    r"""Two interleaving crescents.

    Args:
        key (Any): Random key generator.
        n (int): Number of samples to generate.

    Returns:
        An array of sample points from the two interleaving crescents.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    n1, n2 = n // 2, n - n // 2
    t1 = jax.random.uniform(k1, (n1,)) * jnp.pi
    t2 = jax.random.uniform(k2, (n2,)) * jnp.pi
    noise = jax.random.normal(k3, (n, 2)) * 0.08
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
    k1, k2 = jax.random.split(key)
    t = 1.5 * jnp.pi * (1 + 2 * jax.random.uniform(k1, (n,)))
    noise = jax.random.normal(k2, (n, 2)) * 0.08
    return jnp.stack([t * jnp.cos(t), t * jnp.sin(t)], -1) / 8 + noise


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
        dtype: typing.Any = None,
        param_dtype: typing.Any = None,
        precision: typing.Any = None,
    ) -> None:
        self._dtype = dtype
        self._param_dtype = param_dtype
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


def main(argv: typing.List[str]) -> int:
    del argv  # unused arguments

    key = jrnd.PRNGKey(flags.FLAGS.seed)

    # building model
    _logging.rank_zero_info("Building model...")
    key, init_key = jrnd.split(key, num=2)
    model = MeanFlowMLPModel(
        features=flags.FLAGS.hidden_size,
        num_layers=flags.FLAGS.num_layers,
    )
    params, _ = model.init(
        batch=jnp.zeros((1, 2)),
        rngs=init_key,
    )
    _logging.rank_zero_info("Building model... DONE!")

    _logging.rank_zero_info("Building train state...")
    tx = optax.adam(learning_rate=flags.FLAGS.lr)
    state = _train_state.TrainState.create(
        params=params,
        tx=tx,
        ema_rate=flags.FLAGS.ema_rate,
    )
    state = jax.block_until_ready(state)
    _logging.rank_zero_info("Building train state... DONE!")

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
