import os
import typing

from absl import app
from absl import flags
import jax
from jax import numpy as jnp
from matplotlib import markers as mpl_markers
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# Configure plotting style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Open Sans",
            "Helvetica",
            "DejaVu Sans",
            "sans-serif",
        ],
    }
)

# Flags
flags.DEFINE_float(
    name="buffer_radius",
    default=None,
    required=False,
    help=(
        "Radius around the conditional paths for heatmap evaluation. "
        "If `None`, evaluate the full area."
    ),
)
flags.DEFINE_integer(
    name="n_grid_points",
    default=200,
    help="Number of grid points per axis to evaluate the marginal velocity",
)
flags.DEFINE_integer(
    name="n_samples",
    default=100,
    help="Number of Monte Carlo samples to evaluate the marginal velocity",
)
flags.DEFINE_boolean(
    name="show_samples",
    default=False,
    help="Whether to show the conditional samples xt on the plot.",
)
flags.DEFINE_list(
    name="t_evals",
    default=["0.25", "0.5", "0.75"],
    required=False,
    help="Timesteps at which the velocity field difference heatmap to plot.",
)
flags.DEFINE_integer(
    name="seed",
    default=42,
    help="Random seed for reproducibility.",
)
flags.DEFINE_string(
    name="work_dir",
    default=None,
    required=True,
    help="Output directory.",
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
        [[5.0, 2.0], [4.0, 4.0], [2.0, 5.0]],
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
) -> typing.Callable[[jax.Array, float], typing.Tuple[jax.Array, jax.Array]]:
    r"""Returns Monte-Carlo estimate of the marginal velocity field and the expected difference."""

    def velocity(x: jax.Array, t: float) -> typing.Tuple[jax.Array, jax.Array]:
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

        # compute expected difference: E_{x_{0}|x_{t}} [|| v_marg - v_cond ||]
        diffs = out[:, None, :] - cond_vf[None, :, :]  # Shape: (M, N, 2)
        norms = jnp.linalg.norm(diffs, axis=-1)  # Shape: (M, N)

        # Weighted sum over the N paths
        expected_diff = jnp.sum(weights * norms, axis=-1)  # Shape: (M,)

        return out, expected_diff

    return velocity


# Main entry point
def main(argv: typing.List[str]) -> int:
    del argv  # unused

    rng = jax.random.PRNGKey(flags.FLAGS.seed)
    rng, sample_key = jax.random.split(rng)
    z, data = _create_samples(sample_key, flags.FLAGS.n_samples)

    grid_min, grid_max = -1.0, 8.0
    xs = jnp.linspace(grid_min, grid_max, num=flags.FLAGS.n_grid_points)
    ys = jnp.linspace(grid_min, grid_max, num=flags.FLAGS.n_grid_points)
    X, Y = jnp.meshgrid(xs, ys)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    velocity_fn = _mc_marginal_velocity(z, data)

    # Setup the figure and GridSpec
    fig = plt.figure(figsize=(18, 12))
    fig.set_facecolor("#000000")
    gs = gridspec.GridSpec(
        2, 3, height_ratios=[1, 1.3], hspace=0.3, wspace=0.25
    )

    # ---------------------------------------------------------
    # TOP ROW: Expected difference curve over t (Evaluated on paths)
    # ---------------------------------------------------------
    ax_top = fig.add_subplot(gs[0, :])

    t_steps = jnp.linspace(0.01, 0.99, 50)

    # NOTE: Evaluate strictly along the conditional paths
    def eval_path_mean(t):
        x_t = (1 - t) * data + t * z
        _, exp_diff = velocity_fn(x_t, t)
        return jnp.mean(exp_diff)

    mean_diffs = jax.vmap(eval_path_mean)(t_steps)

    ax_top.plot(t_steps, mean_diffs, color="#c9e5c6", lw=3)

    ax_top.set_facecolor("#000000")
    ax_top.tick_params(colors="white")
    for spine in ax_top.spines.values():
        spine.set_edgecolor("white")
    ax_top.set_xlim(0, 1)
    ax_top.set_xlabel("Flow Time Step (t)", color="white", fontsize=12)
    ax_top.set_ylabel("Mean Expected Difference", color="white", fontsize=12)
    ax_top.set_title(
        "Expected Difference Evaluated Exclusively Along Conditional Paths",
        color="white",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax_top.grid(True, color="#333333", linestyle="--", alpha=0.7)

    # ---------------------------------------------------------
    # BOTTOM ROW: Heatmaps with Current States x_t overlaid
    # ---------------------------------------------------------
    for i, t_eval in enumerate(flags.FLAGS.t_evals):
        t_eval = float(t_eval)
        ax = fig.add_subplot(gs[1, i])

        # evaluate velocity and expected difference on the grid
        _, expected_diff = velocity_fn(grid_points, t_eval)
        diff_heatmap = expected_diff.reshape(X.shape)

        # masking logic to only evaluate differences near conditional paths
        x_t_eval = (1 - t_eval) * data + t_eval * z

        # evaluate distances with shape: (n_grid_points**2, n_samples)
        if flags.FLAGS.buffer_radius is not None:
            dists = jnp.linalg.norm(
                grid_points[:, None, :] - x_t_eval[None, :, :],
                axis=-1,
            )
            min_dists = jnp.min(dists, axis=-1)

            # 3. Create a boolean mask and apply it
            mask = min_dists > flags.FLAGS.buffer_radius
            diff_heatmap_masked = jnp.where(
                mask.reshape(X.shape),
                jnp.nan,
                diff_heatmap,
            )
        else:
            diff_heatmap_masked = diff_heatmap

        # Plot the expected difference heatmap
        mesh = ax.imshow(
            diff_heatmap_masked,
            extent=[grid_min, grid_max, grid_min, grid_max],
            origin="lower",
            cmap="magma",
            alpha=0.95,
            zorder=1,
        )
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(
            r"$\mathbb{E}_{x_0|x_t} [\|v(x_t,t) - v(x_t,t \mid x_0)\|]$",
            color="white",
        )
        cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

        # Plot conditional velocity trajectories
        ax.plot(
            [z[0, 0], data[0, 0]],
            [z[0, 1], data[0, 1]],
            color="#9d9795",
            lw=0.5,
            alpha=0.5,
            zorder=3,
            label=r"Conditional Paths $p(\mathbf{x},t \mid \mathbf{x}_0)$",
        )
        ax.plot(
            [z[1:, 0], data[1:, 0]],
            [z[1:, 1], data[1:, 1]],
            color="#9d9795",
            lw=0.5,
            alpha=0.5,
            zorder=3,
        )

        # Plot endpoints
        ax.scatter(
            z[:, 0],
            z[:, 1],
            fc="#a8d3e0",
            ec="#000000",
            s=40,
            lw=0.0,
            alpha=0.8,
            zorder=4,
            label=r"Latent $\mathbf{x}_1$",
        )
        ax.scatter(
            data[:, 0],
            data[:, 1],
            fc="#f1c6d1",
            ec="#000000",
            s=40,
            lw=0.0,
            alpha=0.8,
            zorder=4,
            label=r"Data $\mathbf{x}_0$",
        )

        # plot the specific current states x_t
        if flags.FLAGS.show_samples:
            ax.scatter(
                x_t_eval[:, 0],
                x_t_eval[:, 1],
                fc="#f4e0b0",
                ec="#000000",
                marker=mpl_markers.MarkerStyle("^", fillstyle="full"),
                s=40,
                lw=0.5,
                alpha=0.6,
                zorder=5,
                label=r"Current State $\mathbf{x}_t$",
            )

        # formatting
        ax.set_aspect("equal")
        ax.set_facecolor("#000000")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.set_xlim(grid_min, grid_max)
        ax.set_ylim(grid_min, grid_max)

        if i == 0:
            ax.set_ylabel("Spatial Dimension 2", fontsize=12, color="white")
        ax.legend(loc="upper right", framealpha=0.95, edgecolor="black")
        ax.set_xlabel("Spatial Dimension 1", fontsize=12, color="white")
        ax.set_title(
            rf"$t={t_eval}$",
            color="white",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

    fig.savefig(
        os.path.join(flags.FLAGS.work_dir, "illustration.pdf"),
        dpi=300,
    )

    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
