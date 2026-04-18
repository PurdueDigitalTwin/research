"""Pre-download ImageNet-1k to local HuggingFace cache."""

import os

import datasets


def main():
    """Download ImageNet-1k dataset to local HuggingFace cache."""
    print("Downloading ImageNet-1k (non-streaming)...")
    ds = datasets.load_dataset(
        "ILSVRC/imagenet-1k",
        token=os.getenv("HF_TOKEN", None),
        revision="49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
        streaming=False,
        num_proc=4,
    )
    print("Download complete.")
    for k, v in ds.items():
        print(f"  {k}: {len(v)} examples")


if __name__ == "__main__":
    main()
