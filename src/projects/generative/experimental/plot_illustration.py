import math
import typing

from absl import app
from absl import flags
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt

# Configure plotting style
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "serif"],
    }
)

# Flags
flags.DEFINE_integer(
    "n_samples",
    default=1_000,
    help="Number of Monte Carlo samples to evaluate the marginal velocity",
)
flags.DEFINE_integer(
    "seed",
    default=42,
    help="Random seed for reproducibility.",
)


def _create_samples(
    key: typing.Any,
    n_samples: int,
) -> typing.Tuple[jax.Array, jax.Array]:
    r"""Samples latent and data points."""
    x_key, comp_key, z_key = jax.random.split(key, num=3)
    z = 0.3 * jax.random.normal(z_key, shape=(n_samples, 2))

    _indices = jax.random.choice(
        comp_key,
        a=3,
        shape=(n_samples,),
        p=jnp.array([0.2, 0.5, 0.3]),
    )
    _modes = jnp.array(
        [[6.0, 6.0], [3.0, 3 * math.sqrt(3.0)], [3 * math.sqrt(3.0), 3.0]],
        dtype=jnp.float32,
    )
    _stddevs = jnp.array([0.2, 0.6, 0.3], dtype=jnp.float32)
    data = jnp.add(
        _modes[_indices],
        _stddevs[_indices][:, None]
        * jax.random.normal(x_key, shape=(n_samples, 2)),
    )

    return z, data


# Main entry point
def main(argv: typing.List[str]) -> int:
    del argv  # unused

    rng = jax.random.PRNGKey(flags.FLAGS.seed)
    rng, sample_key = jax.random.split(rng)
    z, data = _create_samples(sample_key, flags.FLAGS.n_samples)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        z[:, 0],
        z[:, 1],
        fc="#a8d3e0",
        ec="#000000",
        s=20,
        lw=0.0,
        alpha=0.8,
        zorder=5,
        label=r"Latent samples $\mathbf{x}_1\sim\mathcal{N}(0, \mathbf{I})$",
    )
    ax.scatter(
        data[:, 0],
        data[:, 1],
        fc="#f1c6d1",
        ec="#000000",
        s=20,
        lw=0.0,
        alpha=0.8,
        zorder=5,
        label=r"Data samples $\mathbf{x}_0\sim p_{data}$",
    )

    ax.legend(loc="bottom right", framealpha=0.95, edgecolor="black")

    plt.show()
    return 0


if __name__ == "__main__":
    app.run(main=main)
