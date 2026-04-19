"""Pre-compute and cache FID reference statistics for ImageNet.

Usage (on a TPU/GPU worker):
    python src/projects/generative/tools/precompute_fid_stats.py \
        --output gs://pdt_gen_ai/juanwu/cache/imagenet-1k-fid-ref-stats.npz

This downloads the full ImageNet training split, extracts InceptionV3
pool3 features from all ~1.28M images, and writes the mean / covariance
as a .npz file.  Subsequent FID evaluations can load from this cache
instead of repeating the 2-hour computation.
"""

import argparse
import functools
import os

from src.projects.generative.tools import fid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default=(
            "gs://pdt_gen_ai/juanwu/cache"
            "/imagenet-1k-fid-ref-stats.npz"
        ),
        help="Output path for the cached .npz file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for InceptionV3 feature extraction.",
    )
    args = parser.parse_args()

    import datasets

    dataset_fn = functools.partial(
        datasets.load_dataset,
        path="ILSVRC/imagenet-1k",
        token=os.getenv("HF_TOKEN", None),
        revision="49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
        split="train",
    )

    # This triggers download, feature extraction, and cache save.
    fid.FrechetInceptionDistance(
        dataset=dataset_fn,
        image_key="image",
        batch_size=args.batch_size,
        ref_cache_path=args.output,
    )
    print(f"Done. Cached stats written to {args.output}")


if __name__ == "__main__":
    main()
