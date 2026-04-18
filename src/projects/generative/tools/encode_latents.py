"""Pre-encode ImageNet images to SD-VAE latent space.

Encodes raw ImageNet images through a frozen Stable Diffusion
VAE encoder and saves ``(latent_mean, latent_logvar, label)``
as ``.npz`` shard files.  This avoids running the expensive
VAE encoder inside the training loop every step.

Usage::

    bazelisk run --config=tpu \
        //src/projects/generative/tools:encode_latents -- \
        --output_dir gs://pdt_gen_ai/juanwu/cache/imagenet-1k-latent \
        --split train
"""

import functools
import io
import os
import typing

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from src.data import huggingface
from src.data import preprocess
from src.projects.generative.model import vae as _vae

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "vae_path",
    os.getenv("VAE_PATH", "pcuenq/sd-vae-ft-mse-flax"),
    "Path to pretrained VAE weights.",
)
flags.DEFINE_string(
    "data_dir",
    os.getenv(
        "IMAGENET_DATA_DIR",
        "gs://pdt_gen_ai/juanwu/cache/huggingface/imagenet-1k",
    ),
    "GCS or local directory containing ImageNet parquets.",
)
flags.DEFINE_string(
    "output_dir",
    "gs://pdt_gen_ai/juanwu/cache/imagenet-1k-latent",
    "Output directory for latent .npz shards.",
)
flags.DEFINE_integer(
    "batch_size_per_device", 32, "Batch size per TPU/GPU chip."
)
flags.DEFINE_integer("shard_size", 10_000, "Number of samples per .npz shard.")
flags.DEFINE_string("split", "train", "Dataset split to encode.")
flags.DEFINE_bool(
    "distributed",
    False,
    "Enable multi-host distributed mode (required for TPU v4-32).",
)


def _encode_batch(
    vae: _vae.AutoencoderKL,
    params: typing.Any,
    images: jax.Array,
) -> typing.Tuple[jax.Array, jax.Array]:
    """Encode a batch of images to latent (mean, logvar).

    Args:
        vae: AutoencoderKL module.
        params: Frozen VAE parameters.
        images: float32 images in ``[0, 1]``,
            shape ``(B, 256, 256, 3)``.

    Returns:
        Tuple of ``(mean, logvar)`` in float16.
    """
    images = images * 2.0 - 1.0  # [0,1] -> [-1,1]
    mean, logvar = vae.apply({"params": params}, images, method=vae.encode)
    return mean.astype(jnp.float16), logvar.astype(jnp.float16)


def _save_shard(
    output_dir: str,
    split: str,
    shard_idx: int,
    total_shards: int,
    latent_mean: np.ndarray,
    latent_logvar: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Save one .npz shard to local disk or GCS."""
    filename = f"{split}-{shard_idx:05d}-of-{total_shards:05d}.npz"
    path = os.path.join(output_dir, filename)

    buf = io.BytesIO()
    np.savez(
        buf,
        latent_mean=latent_mean,
        latent_logvar=latent_logvar,
        label=labels,
    )
    buf.seek(0)

    if path.startswith("gs://"):
        tf.io.gfile.makedirs(output_dir)
        with tf.io.gfile.GFile(path, "wb") as f:
            f.write(buf.read())
    else:
        os.makedirs(output_dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf.read())

    logging.info("Saved shard %s (%d samples)", path, len(labels))


def main(argv: typing.Sequence[str]) -> None:
    """Entry point."""
    del argv

    if FLAGS.distributed:
        jax.distributed.initialize()

    process_id = jax.process_index()
    num_processes = jax.process_count()
    num_devices = jax.local_device_count()
    batch_per_device = FLAGS.batch_size_per_device
    global_batch = num_devices * batch_per_device
    logging.info(
        "Process %d/%d: %d local devices, %d batch/dev, " "%d local batch",
        process_id,
        num_processes,
        num_devices,
        batch_per_device,
        global_batch,
    )

    # Load VAE
    logging.info("Loading VAE from %s ...", FLAGS.vae_path)
    vae, vae_params = _vae.AutoencoderKL.from_pretrained(FLAGS.vae_path)
    vae_params_replicated = jax.device_put_replicated(
        vae_params, jax.local_devices()
    )
    logging.info("VAE loaded and replicated.")

    # pmap across local devices only (not global)
    p_encode = jax.pmap(
        functools.partial(_encode_batch, vae),
        axis_name="batch",
        devices=jax.local_devices(),
    )

    # Load ImageNet data using existing pipeline
    data = huggingface.ImageNet1KDataModule(
        batch_size=global_batch,
        deterministic=True,
        drop_remainder=False,
        num_workers=4,
        shuffle_buffer_size=1,  # no shuffle for encoding
        streaming=True,
        transform=preprocess.chain(
            functools.partial(
                preprocess.filter_keys,
                keys=["image", "label"],
            ),
            functools.partial(preprocess.resize, size=(256, 256)),
            functools.partial(
                preprocess.normalize,
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
            ),
        ),
        data_dir=FLAGS.data_dir,
    )

    if FLAGS.split == "train":
        loader = data.train_dataloader()
    elif FLAGS.split in ("validation", "val"):
        loader = data.eval_dataloader()
    else:
        raise ValueError(f"Unknown split: {FLAGS.split}")

    # Encode and accumulate shards
    shard_means: typing.List[np.ndarray] = []
    shard_logvars: typing.List[np.ndarray] = []
    shard_labels: typing.List[np.ndarray] = []
    shard_count = 0
    sample_count = 0
    shard_idx = 0

    for batch_idx, batch in enumerate(loader):
        # In multi-host mode, each process handles every Nth batch
        if batch_idx % num_processes != process_id:
            continue

        images = np.asarray(batch["image"])  # [B, 256, 256, 3]
        labels = np.asarray(batch["label"])  # [B]

        actual_b = images.shape[0]
        # Pad to global_batch if needed (last batch)
        if actual_b < global_batch:
            pad_n = global_batch - actual_b
            images = np.concatenate(
                [images, np.zeros_like(images[:pad_n])], axis=0
            )
            labels_padded = np.concatenate(
                [labels, np.zeros(pad_n, dtype=labels.dtype)],
                axis=0,
            )
        else:
            labels_padded = labels

        # Reshape for pmap: [num_devices, batch_per_device, ...]
        images = images.reshape(num_devices, batch_per_device, 256, 256, 3)
        images = jnp.asarray(images)

        mean, logvar = p_encode(vae_params_replicated, images)
        # [num_devices, batch_per_device, 32, 32, 4] -> [B, ...]
        mean = np.asarray(mean).reshape(-1, 32, 32, 4)[:actual_b]
        logvar = np.asarray(logvar).reshape(-1, 32, 32, 4)[:actual_b]
        labels_out = np.asarray(labels_padded)[:actual_b]

        shard_means.append(mean)
        shard_logvars.append(logvar)
        shard_labels.append(labels_out)
        shard_count += actual_b
        sample_count += actual_b

        if batch_idx % (50 * num_processes) < num_processes:
            logging.info(
                "[P%d] Encoded %d samples (%d batches)",
                process_id,
                sample_count,
                batch_idx + 1,
            )

        # Flush shard when we hit shard_size
        while shard_count >= FLAGS.shard_size:
            all_means = np.concatenate(shard_means, axis=0)
            all_logvars = np.concatenate(shard_logvars, axis=0)
            all_labels = np.concatenate(shard_labels, axis=0)

            # Interleave shard indices across processes
            global_shard_idx = shard_idx * num_processes + process_id
            _save_shard(
                output_dir=FLAGS.output_dir,
                split=FLAGS.split,
                shard_idx=global_shard_idx,
                total_shards=99999,
                latent_mean=all_means[: FLAGS.shard_size],
                latent_logvar=all_logvars[: FLAGS.shard_size],
                labels=all_labels[: FLAGS.shard_size],
            )
            shard_idx += 1

            shard_means = [all_means[FLAGS.shard_size :]]
            shard_logvars = [all_logvars[FLAGS.shard_size :]]
            shard_labels = [all_labels[FLAGS.shard_size :]]
            shard_count = len(shard_means[0])

    # Flush remaining samples
    if shard_count > 0:
        all_means = np.concatenate(shard_means, axis=0)
        all_logvars = np.concatenate(shard_logvars, axis=0)
        all_labels = np.concatenate(shard_labels, axis=0)
        global_shard_idx = shard_idx * num_processes + process_id
        _save_shard(
            output_dir=FLAGS.output_dir,
            split=FLAGS.split,
            shard_idx=global_shard_idx,
            total_shards=99999,
            latent_mean=all_means,
            latent_logvar=all_logvars,
            labels=all_labels,
        )
        shard_idx += 1

    logging.info(
        "[P%d] Done! Encoded %d samples into %d shards at %s",
        process_id,
        sample_count,
        shard_idx,
        FLAGS.output_dir,
    )
    return 0


if __name__ == "__main__":
    app.run(main)
