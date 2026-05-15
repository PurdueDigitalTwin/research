"""Pre-compute and cache FID reference statistics for ImageNet.

Usage (on a multi-host TPU, run on all workers):
    bazelisk run --config=tpu \
        //src/projects/generative/tools:precompute_fid_stats -- \
        --distributed=true

Downloads ImageNet train-split parquet files one at a time from
HuggingFace Hub, extracts InceptionV3 pool3 features, and saves the
reference mean/covariance as a ``.npz`` on GCS.  Deletes each
parquet from the local cache after processing to stay within disk
limits.  Checkpoints intermediate stats to GCS every 10 files so
crashes don't lose progress.

Only process 0 does actual work; other processes poll for completion.
"""

import io
import os

from absl import app
from absl import flags
from flax import serialization
import huggingface_hub as hf_hub
import jax
from jax import numpy as jnp
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import tensorflow as tf
from tqdm import auto as tqdm

from src.core import distributed as _distributed
from src.projects.generative.model import inception

flags.DEFINE_bool(
    name="distributed",
    default=False,
    help="Enable multi-host JAX distributed initialization.",
)

_OUTPUT = "gs://pdt_training/juanwu/cache/imagenet-1k-fid-ref-stats.npz"
_CHECKPOINT = (
    "gs://pdt_training/juanwu/cache/" "imagenet-1k-fid-ref-stats-ckpt.npz"
)
_BATCH_SIZE = 64
_FEAT_DIM = 2048
_CKPT_EVERY = 10


def _build_inception():
    """Build InceptionV3 and load weights."""
    model = inception.InceptionV3(
        num_classes=1_008,
        last_block_max_pool=True,
        with_aux_logits=False,
    )
    weights_path = hf_hub.hf_hub_download(
        repo_id="ChocolateDave/fid-inception-v3",
        filename="fid_inception_v3.msgpack",
        token=os.getenv("HF_TOKEN", None),
        revision="bef27900b6b2c46b866b628a86a1c1cedd95a041",
    )
    with open(weights_path, "rb") as f:
        variables = serialization.msgpack_restore(f.read())

    @jax.jit
    def extract(batch: jax.Array):
        inputs = (jnp.astype(batch, jnp.float32) - 128.0) / 128.0
        feat, _ = model.apply(
            variables=dict(  # type: ignore
                params=variables["params"],
                batch_stats=variables["batch_stats"],
            ),
            inputs=inputs,
            deterministic=True,
            with_head=False,
        )
        return feat

    return extract


def _resize_batch_pil(images, size=299):
    """Resize a list of PIL images to (size, size, 3) using PIL."""
    out = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        out.append(np.asarray(img, dtype=np.uint8))
    return np.stack(out, axis=0)


def _list_train_parquets():
    """List all train-split parquet file paths in the repo."""
    token = os.getenv("HF_TOKEN", None)
    files = []
    for entry in hf_hub.list_repo_tree(
        repo_id="ILSVRC/imagenet-1k",
        path_in_repo="data",
        repo_type="dataset",
        token=token,
        revision="49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
    ):
        assert isinstance(entry, hf_hub.RepoFile)
        name = entry.rfilename
        if name.startswith("data/train-") and name.endswith(".parquet"):
            files.append(name)
    files.sort()
    return files


def main(argv) -> None:
    del argv
    if flags.FLAGS.distributed:
        _distributed.setup_jax_distributed()

    # Only process 0 does actual work; others poll for completion.
    if jax.process_index() != 0:
        import time

        print(
            f"Process {jax.process_index()}: "
            "waiting for process 0 to finish..."
        )
        while not tf.io.gfile.exists(_OUTPUT):
            time.sleep(30)
        print(
            f"Process {jax.process_index()}: "
            "cache file detected, shutting down."
        )
        if flags.FLAGS.distributed:
            jax.distributed.shutdown()
        return

    token = os.getenv("HF_TOKEN", None)
    print("Building InceptionV3...")
    extract_fn = _build_inception()

    print("Listing train parquet files...")
    parquet_files = _list_train_parquets()
    print(f"Found {len(parquet_files)} parquet files.")

    # resume from checkpoint if available
    n = 0
    sum_f = np.zeros(_FEAT_DIM, dtype=np.float64)
    sum_ff = np.zeros((_FEAT_DIM, _FEAT_DIM), dtype=np.float64)
    start_idx = 0
    try:
        with tf.io.gfile.GFile(_CHECKPOINT, "rb") as fh:
            ckpt = np.load(io.BytesIO(fh.read()))
            n = int(ckpt["n"])
            sum_f = ckpt["sum_f"]
            sum_ff = ckpt["sum_ff"]
            start_idx = int(ckpt["next_idx"])
        print(
            f"Resumed from checkpoint: {start_idx}/{len(parquet_files)}"
            f" files done, {n} images processed."
        )
    except tf.errors.NotFoundError:
        pass

    pbar = tqdm.tqdm(
        enumerate(parquet_files),
        total=len(parquet_files),
        initial=start_idx,
        desc="Processing parquets",
        unit="file",
    )

    for file_idx, parquet_path in pbar:
        if file_idx < start_idx:
            continue

        # download this single parquet
        local_path = hf_hub.hf_hub_download(
            repo_id="ILSVRC/imagenet-1k",
            revision="49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
            filename=parquet_path,
            repo_type="dataset",
            token=token,
        )

        # read the parquet and extract images
        table = pq.read_table(local_path, columns=["image"])
        image_col = table.column("image")
        images = []

        for row in image_col:
            row_dict = row.as_py()
            img_bytes = row_dict.get("bytes", None)
            if img_bytes is None:
                continue
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)

        # process in batches
        for i in range(0, len(images), _BATCH_SIZE):
            batch_imgs = images[i : i + _BATCH_SIZE]
            batch = _resize_batch_pil(batch_imgs)
            feats = np.asarray(extract_fn(jnp.array(batch))).astype(np.float64)
            n += feats.shape[0]
            sum_f += feats.sum(axis=0)
            sum_ff += feats.T @ feats

        pbar.set_postfix(images=n)
        del table, image_col, images

        # delete cached parquet to free disk space
        try:
            real = os.path.realpath(local_path)
            if os.path.exists(real):
                os.remove(real)
            if os.path.islink(local_path):
                os.remove(local_path)
        except OSError:
            pass

        # checkpoint every _CKPT_EVERY files
        if (file_idx + 1) % _CKPT_EVERY == 0:
            buf = io.BytesIO()
            np.savez(
                buf,
                n=np.array(n),
                sum_f=sum_f,
                sum_ff=sum_ff,
                next_idx=np.array(file_idx + 1),
            )
            buf.seek(0)
            with tf.io.gfile.GFile(_CHECKPOINT, "wb") as fh:
                fh.write(buf.read())
            print(
                f"\nCheckpointed at file {file_idx + 1}/"
                f"{len(parquet_files)}, {n} images."
            )

    mu = sum_f / n
    cov = (sum_ff - n * np.outer(mu, mu)) / (n - 1)
    print(f"Processed {n} images total.")
    print(f"mu shape: {mu.shape}, cov shape: {cov.shape}")

    # save final result to GCS
    buf = io.BytesIO()
    np.savez(buf, mu=mu, cov=cov)
    buf.seek(0)
    with tf.io.gfile.GFile(_OUTPUT, "wb") as fh:
        fh.write(buf.read())
    print(f"Saved to {_OUTPUT}")

    # clean up checkpoint
    try:
        tf.io.gfile.remove(_CHECKPOINT)
    except tf.errors.NotFoundError:
        pass

    if flags.FLAGS.distributed:
        jax.distributed.shutdown()


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
