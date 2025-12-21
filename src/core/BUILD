load("//third_party:defs.bzl", "ml_py_library")

package(default_visibility = ["//visibility:public"])

ml_py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "fiddle",
        "optax",
        ":datamodule",
        ":model",
    ],
)

ml_py_library(
    name = "datamodule",
    srcs = ["datamodule.py"],
)

ml_py_library(
    name = "distributed",
    srcs = ["distributed.py"],
    deps = ["jax"],
)

ml_py_library(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        "jax",
        "jaxtyping",
        "wandb",
        ":datamodule",
        ":model",
        "//src/utilities:logging",
    ],
)

ml_py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "chex",
        "flax",
        "jax",
        "jaxtyping",
    ],
)

ml_py_library(
    name = "train_state",
    srcs = ["train_state.py"],
    deps = [
        "flax",
        "jax",
        "jaxtyping",
        "optax",
    ],
)

ml_py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        "flax",
        "jax",
        "jaxtyping",
        "numpy",
        "orbax-checkpoint",
        "tqdm",
        "wandb",
        ":datamodule",
        ":model",
        ":train_state",
        "//src/utilities:logging",
    ],
)
