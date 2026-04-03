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
    "n_grid_points",
    default=25,
    help="Number of grid points per axis to evaluate the marginal velocity",
)
flags.DEFINE_integer(
    "n_samples",
    default=100,
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


def _mc_marginal_velocity(
    z: jax.Array,
    data: jax.Array,
) -> typing.Callable[[jax.Array, float], jax.Array]:
    r"""Returns Monte-Carlo estimate of the marginal velocity field.

    Args:
        z (jax.Array): Latent samples of shape ``(n_samples, 2)``.
        data (jax.Array): Data samples of shape ``(n_samples, 2)``.

    Returns:
        A function that takes in a point ``x`` of shape ``(num_points, 2)`` and
        a scalar time ``t`` and returns the Monte-Carlo estimate of the
        marginal velocity at that point and time, of shape ``(num_points, 2,)``.
    """

    def velocity(x: jax.Array, t: float) -> jax.Array:
        # compute conditional velocity at x_{t} | x_{0}
        cond_vf = z - data

        # compute log-density p(x_{t}|x_{0},x_{1})=N((1-t)x_{0}+tx_{1},sigma^2I)
        mu = (1 - t) * data + t * z
        log_w = (
            -0.5
            * jnp.sum(
                jnp.square(x[:, None, :] - mu[None, :, :]),
                axis=-1,
            )
            / 0.1
        )
        weights = jax.nn.softmax(log_w, axis=-1)

        # compute marginal velocity v(x_{t},t)=E_{x_{0}|x_{t}}[v(x_{t}|x_{0})]
        out = jnp.einsum("mn,nd->md", weights, cond_vf)

        return out

    return velocity


# Main entry point
def main(argv: typing.List[str]) -> int:
    del argv  # unused

    rng = jax.random.PRNGKey(flags.FLAGS.seed)
    rng, sample_key = jax.random.split(rng)
    z, data = _create_samples(sample_key, flags.FLAGS.n_samples)

    # create a grid of points to eavaluate the marginal velocity field
    xs = jnp.linspace(-1.5, 6.0, num=flags.FLAGS.n_grid_points)
    ys = jnp.linspace(-1.5, 6.0, num=flags.FLAGS.n_grid_points)
    X, Y = jnp.meshgrid(xs, ys)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    velocity_fn = _mc_marginal_velocity(z, data)
    mvf = velocity_fn(grid_points, 1.0)

    # plotting
    fig, ax = plt.subplots(figsize=(6, 6))

    # plot samples
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

    # plot marginal velocity field as quiver plot
    scale = 10 * jnp.sqrt(jnp.sum(mvf**2, axis=-1)).max().item()
    ax.quiver(
        grid_points[:, 0],
        grid_points[:, 1],
        -mvf[:, 0],
        -mvf[:, 1],
        color="#FFFFFF",
        alpha=0.5,
        scale=scale,
        width=0.0035,
        headwidth=3,
    )

    ax.set_aspect("equal")
    ax.set_facecolor("#000000")
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="black")

    plt.show()
    return 0


if __name__ == "__main__":
    app.run(main=main)
