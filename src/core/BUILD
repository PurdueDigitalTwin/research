load("//third_party:defs.bzl", "ml_py_library")

package(default_visibility = ["//visibility:public"])

ml_py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "fiddle",
        "optax",
        ":data",
        ":model",
    ],
)

ml_py_library(
    name = "datamodule",
    srcs = ["datamodule.py"],
)

ml_py_library(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        "clu",
        "jax",
        ":data",
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
        "jaxtyping",
        ":train_state",
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
        "clu",
        "flax",
        "jax",
        "jaxtyping",
        ":data",
        ":model",
        ":train_state",
        "//src/utilities:logging",
    ],
)
