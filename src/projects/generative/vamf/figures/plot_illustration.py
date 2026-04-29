"""Conditional-marginal velocity gap illustration."""

import os
import typing

from absl import app
from absl import flags
import jax
from jax import numpy as jnp
from matplotlib import gridspec
from matplotlib import markers as mpl_markers
from matplotlib import pyplot as plt

from src.projects.generative.vamf.figures import _style

# -- flags ---------------------------------------------------------------------
flags.DEFINE_enum(
    "style",
    _style.DEFAULT_STYLE,
    list(_style.STYLES),
    "Render target. 'paper' = light/serif; 'slides' = dark/sans-serif.",
)
flags.DEFINE_float(
    "buffer_radius",
    None,
    "Radius around the conditional paths for heatmap evaluation. "
    "If None, evaluate the full area.",
    required=False,
)
flags.DEFINE_integer(
    "n_grid_points",
    200,
    "Number of grid points per axis to evaluate the marginal velocity.",
)
flags.DEFINE_integer(
    "n_samples",
    100,
    "Number of Monte Carlo samples to estimate the marginal velocity.",
)
flags.DEFINE_boolean(
    "show_samples",
    False,
    "Whether to overlay the conditional samples x_t.",
)
flags.DEFINE_list(
    "t_evals",
    ["0.25", "0.5", "0.75"],
    "Timesteps at which to draw the spatial gap heatmap.",
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string(
    "work_dir",
    None,
    "Output directory.",
    required=True,
)
flags.DEFINE_string(
    "filename",
    "illustration.pdf",
    "Output filename.",
)


# -- helpers -------------------------------------------------------------------
def _create_samples(
    key: typing.Any,
    n_samples: int,
) -> typing.Tuple[jax.Array, jax.Array]:
    """Sample latent and 3-mode mixture data."""
    x_key, comp_key, z_key = jax.random.split(key, num=3)
    z = 0.3 * jax.random.normal(z_key, shape=(n_samples, 2))
    indices = jax.random.choice(
        comp_key,
        a=3,
        shape=(n_samples,),
        p=jnp.array([0.2, 0.5, 0.3]),
    )
    modes = jnp.array(
        [[5.0, 2.0], [4.0, 4.0], [2.0, 5.0]],
        dtype=jnp.float32,
    )
    stddevs = jnp.array([0.2, 0.6, 0.3], dtype=jnp.float32)
    data = modes[indices] + stddevs[indices][:, None] * jax.random.normal(
        x_key,
        shape=(n_samples, 2),
    )
    return z, data


def _mc_marginal_velocity(z, data):
    """Returns Monte-Carlo estimator of v(x,t) and E[||v_cond - v||]."""

    def velocity(x: jax.Array, t: float):
        cond_vf = z - data  # (N, 2)
        mu = (1 - t) * data + t * z  # (N, 2)
        log_w = (
            -0.5
            * jnp.sum(
                jnp.square(x[:, None, :] - mu[None, :, :]),
                axis=-1,
            )
            / 0.1
        )
        weights = jax.nn.softmax(log_w, axis=-1)
        out = jnp.einsum("mn,nd->md", weights, cond_vf)
        diffs = out[:, None, :] - cond_vf[None, :, :]
        norms = jnp.linalg.norm(diffs, axis=-1)
        expected_diff = jnp.sum(weights * norms, axis=-1)
        return out, expected_diff

    return velocity


# -- panels --------------------------------------------------------------------
def _draw_top_panel(ax, t_steps, mean_diffs, palette):
    ax.plot(
        t_steps,
        mean_diffs,
        color=palette["top_curve"],
        linewidth=2.0,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"flow time step $t$")
    ax.set_ylabel(
        r"$\mathbb{E}_{x_0 \mid x_t}\,\|v(x_t,t) - v(x_t,t\mid x_0)\|$"
    )
    ax.set_title(
        "Conditional-marginal velocity gap along the conditional paths",
        loc="left",
        fontweight="bold",
    )


def _draw_heatmap_panel(
    ax: plt.Axes,
    X: jax.Array,
    grid_points: jax.Array,
    t_eval: float,
    z: jax.Array,
    data: jax.Array,
    velocity_fn: typing.Callable,
    grid_min: float,
    grid_max: float,
    buffer_radius: float,
    show_samples: bool,
    palette,
    is_first: bool,
    is_last: bool,
) -> plt.Axes:
    _, expected_diff = velocity_fn(grid_points, t_eval)
    diff_heatmap = expected_diff.reshape(X.shape)

    if buffer_radius is not None:
        x_t_eval = (1 - t_eval) * data + t_eval * z
        dists = jnp.linalg.norm(
            grid_points[:, None, :] - x_t_eval[None, :, :],
            axis=-1,
        )
        min_dists = jnp.min(dists, axis=-1)
        mask = min_dists > buffer_radius
        diff_heatmap = jnp.where(
            mask.reshape(X.shape),
            jnp.nan,
            diff_heatmap,
        )

    mesh = ax.imshow(
        diff_heatmap,
        extent=[grid_min, grid_max, grid_min, grid_max],
        origin="lower",
        cmap=palette["heatmap_cmap"],
        alpha=0.95,
        zorder=1,
    )
    if is_last:
        fig = ax.get_figure()
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(
            r"$\mathbb{E}_{x_0 \mid x_t}\,\|v - v_{\mathrm{cond}}\|$"
        )

    # Conditional paths.
    ax.plot(
        [z[:, 0], data[:, 0]],
        [z[:, 1], data[:, 1]],
        color=palette["path"],
        lw=0.5,
        alpha=0.5,
        zorder=3,
    )
    # Endpoints.
    ax.scatter(
        z[:, 0],
        z[:, 1],
        c=palette["latent"],
        s=22,
        lw=0.0,
        alpha=0.8,
        zorder=4,
        label=r"latent $\mathbf{x}_1$",
    )
    ax.scatter(
        data[:, 0],
        data[:, 1],
        c=palette["data"],
        s=22,
        lw=0.0,
        alpha=0.8,
        zorder=4,
        label=r"data $\mathbf{x}_0$",
    )
    if show_samples:
        x_t_eval = (1 - t_eval) * data + t_eval * z
        ax.scatter(
            x_t_eval[:, 0],
            x_t_eval[:, 1],
            c=palette["state"],
            marker=mpl_markers.MarkerStyle("^", fillstyle="full"),
            s=22,
            lw=0.5,
            alpha=0.7,
            zorder=5,
            label=r"current state $\mathbf{x}_t$",
        )

    ax.set_aspect("equal")
    ax.set_xlim(grid_min, grid_max)
    ax.set_ylim(grid_min, grid_max)
    ax.set_xlabel(r"$x^{(1)}$")
    if is_first:
        ax.set_ylabel(r"$x^{(2)}$")
    else:
        ax.set_ylabel("")
    ax.set_title(
        rf"$t = {t_eval:.2f}$",
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=7)

    return ax


# -- main ----------------------------------------------------------------------
def main(argv: typing.List[str]) -> int:
    del argv
    F = flags.FLAGS
    os.makedirs(F.work_dir, exist_ok=True)

    _style.apply_style(F.style)
    palette = _style.palette(F.style)

    rng = jax.random.PRNGKey(F.seed)
    rng, sample_key = jax.random.split(rng)
    z, data = _create_samples(sample_key, F.n_samples)

    grid_min, grid_max = -1.0, 8.0
    xs = jnp.linspace(grid_min, grid_max, num=F.n_grid_points)
    ys = jnp.linspace(grid_min, grid_max, num=F.n_grid_points)
    X, Y = jnp.meshgrid(xs, ys)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    velocity_fn = _mc_marginal_velocity(z, data)

    fig = plt.figure(figsize=(10.5, 7.5))
    gs = gridspec.GridSpec(
        2,
        3,
        height_ratios=[1, 1.4],
        hspace=0.32,
        wspace=0.28,
        figure=fig,
    )
    ax_top = fig.add_subplot(gs[0, :])

    t_steps = jnp.linspace(0.01, 0.99, 50)

    def eval_path_mean(t):
        x_t = (1 - t) * data + t * z
        _, exp_diff = velocity_fn(x_t, t)
        return jnp.mean(exp_diff)

    mean_diffs = jax.vmap(eval_path_mean)(t_steps)
    _draw_top_panel(ax_top, t_steps, mean_diffs, palette)

    for i, t_str in enumerate(F.t_evals):
        t_eval = float(t_str)
        ax = fig.add_subplot(gs[1, i])
        ax = _draw_heatmap_panel(
            ax,
            X,
            grid_points,
            t_eval,
            z,
            data,
            velocity_fn,
            grid_min,
            grid_max,
            F.buffer_radius,
            F.show_samples,
            palette,
            is_first=(i == 0),
            is_last=(i == len(F.t_evals) - 1),
        )

    out_path = os.path.join(F.work_dir, F.filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}  (style={F.style})")
    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
