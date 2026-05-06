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
    name="style",
    default=_style.DEFAULT_STYLE,
    enum_values=list(_style.STYLES),
    help="Render target. 'paper' = light/serif; 'slides' = dark/sans-serif.",
)
flags.DEFINE_float(
    name="buffer_radius",
    default=None,
    help=(
        "Radius around the conditional paths for heatmap evaluation. "
        "If None, evaluate the full area."
    ),
    required=False,
)
flags.DEFINE_integer(
    name="n_grid_points",
    default=200,
    help="Number of grid points per axis to evaluate the marginal velocity.",
)
flags.DEFINE_integer(
    name="n_samples",
    default=100,
    help="Number of Monte Carlo samples to estimate the marginal velocity.",
)
flags.DEFINE_boolean(
    name="show_samples",
    default=False,
    help="Whether to overlay the conditional samples ``x_t``.",
)
flags.DEFINE_list(
    name="t_evals",
    default=["0.25", "0.5", "0.75"],
    help="Timesteps at which to draw the spatial gap heatmap.",
)
flags.DEFINE_integer(
    name="seed",
    default=42,
    help="Random seed.",
)
flags.DEFINE_string(
    "work_dir",
    None,
    "Output directory.",
    required=True,
)
flags.DEFINE_string(
    name="filename",
    default="illustration.pdf",
    help="Output filename.",
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


def _mc_marginal_velocity(
    z: jax.Array,
    data: jax.Array,
) -> typing.Callable[[jax.Array, float], typing.Tuple[jax.Array, jax.Array]]:
    r"""Returns Monte-Carlo estimator of v(x,t) and ``sqrt(Tr(Sigma_{v'}))``.

    Args:
        z (jax.Array): Latent features with a shape of ``(*, 2)``.
        data (jax.Array): Samples from data distribution of shape ``(*, 2)``.

    Returns:
        A wrapped function which takes in a tuple of location ``x`` and flow
            time step ``t`` and returns a tuple of Monte-Carlo estimator of
            the marginal velocity ``v(x,t)`` and the trace of the covariance
            of the conditional flunctuation.
    """

    def velocity(x: jax.Array, t: float) -> typing.Tuple[jax.Array, jax.Array]:
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
        # Per-pair fluctuation v(x_t,t) - v_{cond,n}
        diffs = out[:, None, :] - cond_vf[None, :, :]  # (M, N, 2)
        sq_norms = jnp.sum(jnp.square(diffs), axis=-1)  # (M, N)
        # Tr(Sigma_{v'} | x_t) = E_{x_0|x_t}[||v'||^2]
        expected_sq = jnp.sum(weights * sq_norms, axis=-1)  # (M,)
        rms_fluct = jnp.sqrt(expected_sq)
        return out, rms_fluct

    return velocity


# -- panels --------------------------------------------------------------------
def _compute_heatmap(
    X: jax.Array,
    grid_points: jax.Array,
    t_eval: float,
    z: jax.Array,
    data: jax.Array,
    velocity_fn: typing.Callable,
    buffer_radius: typing.Optional[float],
) -> jax.Array:
    r"""Return the RMS-fluctuation heatmap for a given single ``t_eval``."""

    _, rms_fluct = velocity_fn(grid_points, t_eval)
    diff_heatmap = rms_fluct.reshape(X.shape)
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
    assert isinstance(diff_heatmap, jax.Array)

    return diff_heatmap


def _draw_heatmap_panel(
    ax: plt.Axes,
    diff_heatmap: jax.Array,
    t_eval: float,
    z: jax.Array,
    data: jax.Array,
    grid_min: float,
    grid_max: float,
    show_samples: bool,
    palette,
    is_first: bool,
    vmin: float,
    vmax: float,
) -> typing.Any:
    r"""Render a single heatmap subplot."""
    mesh = ax.imshow(
        diff_heatmap,
        extent=[grid_min, grid_max, grid_min, grid_max],
        origin="lower",
        cmap=palette["heatmap_cmap"],
        alpha=0.95,
        zorder=1,
        vmin=vmin,
        vmax=vmax,
    )

    # Conditional paths.
    ax.plot(
        [z[:, 0], data[:, 0]],
        [z[:, 1], data[:, 1]],
        color=palette["path"],
        lw=0.5,
        alpha=0.25,
        zorder=3,
    )
    # Endpoints.
    ax.scatter(
        z[:, 0],
        z[:, 1],
        c=palette["latent"],
        s=30,
        lw=0.0,
        alpha=0.5,
        zorder=4,
        label=r"latent $\mathbf{x}_1$",
    )
    ax.scatter(
        data[:, 0],
        data[:, 1],
        c=palette["data"],
        s=30,
        lw=0.0,
        alpha=0.5,
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
        fontsize=8,
    )
    ax.legend(loc="upper right", fontsize=8)

    return mesh


# -- main ----------------------------------------------------------------------
def main(argv: typing.List[str]) -> int:
    del argv  # unused arguments

    F = flags.FLAGS
    os.makedirs(F.work_dir, exist_ok=True)

    _style.apply_style(F.style)
    palette = _style.palette(F.style)

    rng = jax.random.PRNGKey(F.seed)
    rng, sample_key = jax.random.split(rng)
    z, data = _create_samples(sample_key, F.n_samples)

    grid_min, grid_max = -1.5, 7.5
    xs = jnp.linspace(grid_min, grid_max, num=F.n_grid_points)
    ys = jnp.linspace(grid_min, grid_max, num=F.n_grid_points)
    X, Y = jnp.meshgrid(xs, ys)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    velocity_fn = _mc_marginal_velocity(z, data)

    # Pre-compute heatmaps for all t values so panels share the same scale.
    t_evals = [float(s) for s in F.t_evals]
    heatmaps = [
        _compute_heatmap(
            X, grid_points, t, z, data, velocity_fn, F.buffer_radius
        )
        for t in t_evals
    ]
    finite_vals = jnp.concatenate([h.ravel() for h in heatmaps])
    finite_mask = jnp.isfinite(finite_vals)
    vmin = float(jnp.min(jnp.where(finite_mask, finite_vals, jnp.inf)))
    vmax = float(jnp.max(jnp.where(finite_mask, finite_vals, -jnp.inf)))

    fig = plt.figure(figsize=(10.5, 4.0))
    gs = gridspec.GridSpec(1, 3, wspace=0.28, figure=fig)
    axes = []
    mesh = None
    for i, (t_eval, heatmap) in enumerate(zip(t_evals, heatmaps)):
        ax = fig.add_subplot(gs[0, i])
        mesh = _draw_heatmap_panel(
            ax,
            heatmap,
            t_eval,
            z,
            data,
            grid_min,
            grid_max,
            F.show_samples,
            palette,
            is_first=(i == 0),
            vmin=vmin,
            vmax=vmax,
        )
        axes.append(ax)

    # Single colorbar shared across all panels.
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label(r"$\sqrt{\mathrm{Tr}(\Sigma_{v'} \mid x_t)}$", fontsize=10)

    out_path = os.path.join(F.work_dir, F.filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}  (style={F.style})")
    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main=main)
