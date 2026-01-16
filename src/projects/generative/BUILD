load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library")

package(default_visibility = ["//src/projects/generative:__subpackages__"])

ml_py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "datasets",
        "fiddle",
        "optax",
        ":meanflow",
        "//src/core:config",
        "//src/data:huggingface",
        "//src/data:preprocess",
        "//src/projects/generative/tools:fid",
    ],
)

ml_py_library(
    name = "ddpm",
    srcs = ["ddpm.py"],
    deps = [
        "flax",
        "jax",
        "jaxtyping",
    ],
)

ml_py_library(
    name = "experiment",
    srcs = ["experiment.py"],
    deps = [
        "fiddle",
        "jax",
        "jaxtyping",
        "optax",
        "orbax-checkpoint",
        "tensorflow-cpu",
        "tqdm",
        "wandb",
        ":config",
        "//src/core:train",
        "//src/core:train_state",
        "//src/projects/generative/tools:fid",
        "//src/utilities:logging",
        "//src/utilities:visualization",
    ],
)

ml_py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        "absl-py",
        "fiddle",
        "jax",
        ":experiment",
        "//src/core:distributed",
    ],
)

ml_py_library(
    name = "meanflow",
    srcs = ["meanflow.py"],
    deps = [
        "chex",
        "flax",
        "jax",
        "jaxtyping",
        "typing_extensions",
        "//src/core:model",
        "//src/core:train_state",
        "//src/projects/generative/model:unet",
        "//src/projects/generative/pipeline:augment",
    ],
)
