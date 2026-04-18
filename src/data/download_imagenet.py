"""Download ImageNet-1k parquets and upload to GCS.

Usage::

    HF_TOKEN=... bazelisk run --config=tpu //src/data:download_imagenet -- \\
        --gcs_dir gs://pdt_gen_ai/juanwu/cache/huggingface/imagenet-1k
"""

import argparse
import os
import subprocess  # nosec B404

import huggingface_hub


def main():
    """Download ImageNet-1k parquets to GCS for fast streaming."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gcs_dir",
        type=str,
        required=True,
        help="GCS destination for parquet files.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="/dev/shm/data/imagenet-parquets",  # nosec B108
        help="Temporary local directory for downloads.",
    )
    args = parser.parse_args()

    print(f"Downloading parquets to {args.local_dir}...")
    huggingface_hub.snapshot_download(
        repo_id="ILSVRC/imagenet-1k",
        repo_type="dataset",
        revision="49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
        local_dir=args.local_dir,
        allow_patterns="data/*.parquet",
        token=os.getenv("HF_TOKEN", None),
    )
    print("Download complete.")

    src = os.path.join(args.local_dir, "data")
    dst = args.gcs_dir.rstrip("/")
    print(f"Uploading {src} -> {dst}/")
    subprocess.run(  # nosec B603 B607
        ["gcloud", "storage", "rsync", "-r", src, dst],
        check=True,
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
