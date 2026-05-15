"""Generate qualitative samples across DiT-B/4 checkpoints using the same seed.

Usage::

    bazelisk run --config=cuda \\
        //src/projects/generative/vamf/experiments:sample_dit_qualitative -- \\
        --baseline_dir=logs/vamf/dit_checkpoints/baseline \\
        --beta05_dir=logs/vamf/dit_checkpoints/beta05 \\
        --beta1_dir=logs/vamf/dit_checkpoints/beta1 \\
        --classes=9,207,281,387,537,933,974,980 \\
        --seed=42 \\
        --output=src/projects/generative/vamf/report/assets/figures/dit_qualitative.pdf

NOTE: The checkpoint dirs must each contain a ``params/`` subdirectory in
Orbax format (as written by the training pipeline). The model and VAE
configurations are inferred from
``meanflow_dit_imagenet_256_latent`` so the three checkpoints share the
same backbone — only EMA params differ.
"""

import os
import typing

from absl import app
from absl import flags
from etils import epath
import fiddle as fdl
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random as jrnd
import jaxtyping
import matplotlib.pyplot as plt
import numpy as np
from orbax import checkpoint as ocp

from src.projects.generative import config as _cfg
from src.utilities import logging as _logging

# Flags
flags.DEFINE_string(
    name="baseline_dir",
    default=None,
    help="Path to beta=0 checkpoint dir (multi-row mode).",
)
flags.DEFINE_string(
    name="beta05_dir",
    default=None,
    help="Path to beta=0.5 checkpoint dir (multi-row mode).",
)
flags.DEFINE_string(
    name="beta1_dir",
    default=None,
    help="Path to beta=1 checkpoint dir (multi-row mode).",
)
flags.DEFINE_string(
    name="single_dir",
    default=None,
    help=(
        "When set, single-checkpoint mode: sample --n_samples images from this "
        "one checkpoint and lay them out as --grid_rows x --grid_cols."
    ),
)
flags.DEFINE_integer(
    name="grid_rows",
    default=10,
    help="Rows of the grid in single-checkpoint mode.",
)
flags.DEFINE_integer(
    name="grid_cols",
    default=10,
    help="Columns of the grid in single-checkpoint mode.",
)
flags.DEFINE_integer(
    name="batch_size",
    default=0,
    help="Sampling batch size (0 = sample all at once).",
)
flags.DEFINE_list(
    name="classes",
    default=None,
    help=(
        "Optional explicit ImageNet class indices (comma-separated). "
        "If `None`, draw --n_samples random labels using --label_seed."
    ),
)
flags.DEFINE_integer(
    name="n_samples",
    default=12,
    help="Number of samples per row when --classes is not set.",
)
flags.DEFINE_integer(
    name="label_seed",
    default=7,
    help="Seed used to draw random class labels (independent of --seed).",
)
flags.DEFINE_integer(
    name="num_classes",
    default=1000,
    help="Class label upper bound for random sampling.",
)
flags.DEFINE_integer(
    name="seed",
    default=42,
    help="PRNG seed for the noise; shared across all checkpoints.",
)
flags.DEFINE_string(
    name="output",
    default=None,
    help="Output PDF path.",
    required=True,
)
flags.DEFINE_string(
    name="config_fn",
    default="meanflow_dit_imagenet_256_latent",
    help="Config fn name.",
)


def _restore_params(model: nn.Module, checkpoint_dir: str) -> jaxtyping.PyTree:
    r"""Load EMA params from ``<checkpoint_dir>/params`` (Orbax PyTree)."""
    init_rng = jrnd.PRNGKey(0)
    params, _ = model.init(batch=None, rngs=init_rng)
    params_dir = epath.Path(os.path.join(checkpoint_dir.rstrip("/"), "params"))
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(sharding=sharding),
        params,
    )
    handler = ocp.PyTreeCheckpointHandler()
    return handler.restore(
        directory=params_dir,
        item=params,
        transforms=None,
        restore_args=restore_args,
    )


def _sample_one_checkpoint(
    model: nn.Module,
    params: jaxtyping.PyTree,
    classes: typing.Sequence[int],
    seed: int,
    batch_size: int = 0,
) -> np.ndarray:
    r"""Run one-step sampling on the given class list with a fixed seed.

    When ``batch_size`` is 0, all classes are sampled in one forward pass.
    Otherwise, sampling is chunked. Each chunk uses a deterministic
    sub-key derived from ``seed`` so the output is invariant to chunking.

    Args:
        model (nn.Module): DiT model to generate samples from.
        params (PyTree): Restored model parameters.
        classes (Sequence[int]): Class indices to generate samples from.
        seed (int): Random generator seed for sampling noise.
        batch_size (int, optional): Batch size for sampling. If :math:`0`,
            all classes are sampled in one forward pass. Otherwise, sampling is
            chunked and each chunk uses a deterministic sub-key derived from ``seed``. Default is :math:`0`

    Returns:
        A ``(N, H, W, 3)`` numpy array in ``[0, 255]`` with ``dtype=uint8``.
    """
    n = len(classes)
    labels_full = jnp.asarray(classes, dtype=jnp.int32)
    chunks = batch_size if batch_size > 0 else n
    out_chunks = []
    rng_root = jrnd.PRNGKey(seed)
    n_chunks = (n + chunks - 1) // chunks
    sub_keys = (
        jrnd.split(rng_root, n_chunks)
        if n_chunks > 1
        else jnp.stack([rng_root])
    )
    for i, start in enumerate(range(0, n, chunks)):
        end = min(start + chunks, n)
        labels = labels_full[start:end]
        batch = {"label": labels}
        pixel_shape = (end - start, 256, 256, 3)
        out = model.forward(
            rngs=sub_keys[i],
            params=params,
            shape=pixel_shape,
            batch=batch,
            deterministic=True,
        )
        out_chunks.append(np.asarray(out.output))
    img = np.concatenate(out_chunks, axis=0)
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img


def _tile_single_grid(
    imgs: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    out_path: str,
) -> None:
    r"""Render a single grid of N images (N == grid_rows*grid_cols)."""
    n_total = grid_rows * grid_cols
    if imgs.shape[0] < n_total:
        raise ValueError(
            f"need {n_total} samples for {grid_rows}x{grid_cols} grid, got {imgs.shape[0]}"
        )
    cell_w = 0.7
    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(grid_cols * cell_w, grid_rows * cell_w),
        squeeze=False,
    )
    idx = 0
    for i in range(grid_rows):
        for j in range(grid_cols):
            ax = axes[i][j]
            ax.imshow(imgs[idx])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            idx += 1
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _tile_grid(
    rows: typing.Sequence[np.ndarray],
    labels: typing.Sequence[str],
    col_titles: typing.Optional[typing.Sequence[str]],
    out_path: str,
) -> None:
    r"""Compose a ``rows``-by-N grid of images and save as PDF."""
    n_rows = len(rows)
    n_cols = rows[0].shape[0]
    cell_w = 1.1 if n_cols > 8 else 1.6
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * cell_w, n_rows * (cell_w + 0.05) + 0.4),
        squeeze=False,
    )
    for i, row_imgs in enumerate(rows):
        for j in range(n_cols):
            ax = axes[i][j]
            ax.imshow(row_imgs[j])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0 and col_titles is not None:
                ax.set_title(col_titles[j], fontsize=7, pad=2)
        axes[i][0].set_ylabel(
            labels[i],
            fontsize=10,
            rotation=0,
            ha="right",
            va="center",
            labelpad=18,
        )
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main(argv: typing.List[str]) -> int:
    del argv  # unused arguments
    F = flags.FLAGS

    if F.single_dir is not None:
        n_total = F.grid_rows * F.grid_cols
        if F.classes:
            classes = [int(c) for c in F.classes]
            if len(classes) < n_total:
                raise ValueError(
                    f"--classes has {len(classes)} entries but grid needs {n_total}"
                )
            classes = classes[:n_total]
        else:
            rng = np.random.default_rng(F.label_seed)
            classes = rng.integers(
                low=0, high=F.num_classes, size=n_total
            ).tolist()
            _logging.rank_zero_info(
                "Random class labels (seed=%d, %d total): %s",
                F.label_seed,
                n_total,
                classes,
            )

        _logging.rank_zero_info("Building DiT model from %s", F.config_fn)
        config_fn = getattr(_cfg, F.config_fn)
        exp_config = config_fn()
        model = fdl.build(exp_config.model)()
        _logging.rank_zero_info(
            "Backend: %s, devices: %s",
            jax.default_backend(),
            jax.devices(),
        )
        _logging.rank_zero_info(
            "Restoring single checkpoint from %s", F.single_dir
        )
        params = _restore_params(model, F.single_dir)
        _logging.rank_zero_info(
            "Sampling %d images (batch_size=%d)...",
            n_total,
            F.batch_size,
        )
        imgs = _sample_one_checkpoint(
            model,
            params,
            classes,
            F.seed,
            batch_size=F.batch_size,
        )
        _tile_single_grid(imgs, F.grid_rows, F.grid_cols, F.output)
        _logging.rank_zero_info("Saved single grid to %s", F.output)
        return 0

    # Multi-row mode (3 checkpoints, one row each).
    if not (F.baseline_dir and F.beta05_dir and F.beta1_dir):
        raise ValueError(
            "Either set --single_dir, or provide all three of "
            "--baseline_dir/--beta05_dir/--beta1_dir."
        )

    if F.classes:
        classes = [int(c) for c in F.classes]
        col_titles = [f"cls {c}" for c in classes]
    else:
        rng = np.random.default_rng(F.label_seed)
        classes = rng.integers(
            low=0, high=F.num_classes, size=F.n_samples
        ).tolist()
        col_titles = None
        _logging.rank_zero_info(
            "Random class labels (seed=%d): %s", F.label_seed, classes
        )

    _logging.rank_zero_info("Building DiT model from %s", F.config_fn)
    config_fn = getattr(_cfg, F.config_fn)
    exp_config = config_fn()
    model = fdl.build(exp_config.model)()
    _logging.rank_zero_info(
        "Backend: %s, devices: %s", jax.default_backend(), jax.devices()
    )

    rows = []
    row_specs = [
        ("$\\beta\\!=\\!0$", F.baseline_dir),
        ("$\\beta\\!=\\!0.5$", F.beta05_dir),
        ("$\\beta\\!=\\!1$", F.beta1_dir),
    ]
    for label, ckpt in row_specs:
        _logging.rank_zero_info("Restoring %s from %s", label, ckpt)
        params = _restore_params(model, ckpt)
        _logging.rank_zero_info("Sampling %d classes...", len(classes))
        imgs = _sample_one_checkpoint(
            model,
            params,
            classes,
            F.seed,
            batch_size=F.batch_size,
        )
        rows.append(imgs)
        del params
        jax.clear_caches()

    labels = [r[0] for r in row_specs]
    _tile_grid(rows, labels, col_titles, F.output)
    _logging.rank_zero_info("Saved grid to %s", F.output)
    return 0


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
