"""Pre-compute and cache FID reference statistics for ImageNet.

Usage (on a multi-host TPU, run on all workers):
    bazelisk run --config=tpu \
        //src/projects/generative/tools:precompute_fid_stats -- \
        --distributed=true

Downloads ImageNet train-split parquet files one at a time from
HuggingFace Hub, extracts InceptionV3 pool3 features, and saves the
reference mean/covariance as a ``.npz`` on GCS.  Only needs ~1GB of
disk at any time (one parquet at a time).

Only process 0 does actual work; other processes exit after JAX init.
"""

import io
import os
import sys

from absl import app
from absl import flags
from flax import serialization
from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_tree
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

_REPO_ID = "ILSVRC/imagenet-1k"
_REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
_OUTPUT = (
    "gs://pdt_gen_ai/juanwu/cache/imagenet-1k-fid-ref-stats.npz"
)
_BATCH_SIZE = 64
_FEAT_DIM = 2048


def _build_inception():
    """Build InceptionV3 and load weights."""
    model = inception.InceptionV3(
        num_classes=1_008,
        last_block_max_pool=True,
        with_aux_logits=False,
    )
    weights_path = hf_hub_download(
        repo_id="ChocolateDave/fid-inception-v3",
        filename="fid_inception_v3.msgpack",
        token=os.getenv("HF_TOKEN", None),
        revision="bef27900b6b2c46b866b628a86a1c1cedd95a041",
    )
    with open(weights_path, "rb") as f:
        variables = serialization.msgpack_restore(f.read())

    @jax.jit
    def extract(batch):
        inputs = (jnp.astype(batch, jnp.float32) - 128.0) / 128.0
        feat, _ = model.apply(
            variables={
                "params": variables["params"],
                "batch_stats": variables["batch_stats"],
            },
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
    for entry in list_repo_tree(
        _REPO_ID,
        path_in_repo="data",
        repo_type="dataset",
        token=token,
        revision=_REVISION,
    ):
        name = entry.rfilename
        if name.startswith("data/train-") and name.endswith(
            ".parquet"
        ):
            files.append(name)
    files.sort()
    return files


def main(argv) -> None:
    del argv
    if flags.FLAGS.distributed:
        _distributed.setup_jax_distributed()

    # Only process 0 does actual work
    if jax.process_index() != 0:
        print(
            f"Process {jax.process_index()}: "
            "not the primary, exiting."
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

    n = 0
    sum_f = np.zeros(_FEAT_DIM, dtype=np.float64)
    sum_ff = np.zeros((_FEAT_DIM, _FEAT_DIM), dtype=np.float64)

    pbar = tqdm.tqdm(
        parquet_files,
        desc="Processing parquets",
        unit="file",
    )

    for parquet_path in pbar:
        # download this single parquet to a temp file
        local_path = hf_hub_download(
            repo_id=_REPO_ID,
            filename=parquet_path,
            repo_type="dataset",
            token=token,
            revision=_REVISION,
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
            feats = np.asarray(
                extract_fn(jnp.array(batch))
            ).astype(np.float64)
            n += feats.shape[0]
            sum_f += feats.sum(axis=0)
            sum_ff += feats.T @ feats

        pbar.set_postfix(images=n)
        del table, image_col, images

    mu = sum_f / n
    cov = (sum_ff - n * np.outer(mu, mu)) / (n - 1)
    print(f"Processed {n} images total.")
    print(f"mu shape: {mu.shape}, cov shape: {cov.shape}")

    # save to GCS
    buf = io.BytesIO()
    np.savez(buf, mu=mu, cov=cov)
    buf.seek(0)
    with tf.io.gfile.GFile(_OUTPUT, "wb") as fh:
        fh.write(buf.read())
    print(f"Saved to {_OUTPUT}")

    if flags.FLAGS.distributed:
        jax.distributed.shutdown()


if __name__ == "__main__":
    app.run(main)
