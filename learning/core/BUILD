load("//learning:defs.bzl", "ml_py_binary", "ml_py_library")

package(default_visibility = ["//learning:__subpackages__"])

ml_py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        "absl-py",
        "clu",
        "fiddle",
        "flax",
        "tensorflow",
        ":config",
        ":train",
        "//learning:all_config",
        "//learning/utilities:logging",
    ],
)

ml_py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "fiddle",
        ":mixin",
    ],
)

ml_py_library(
    name = "mixin",
    srcs = ["mixin.py"],
    deps = [
        "flax",
        "jax",
        "jaxlib",
        "jaxtyping",
    ],
)

ml_py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        "fiddle",
        "flax",
        "jax",
        "jaxlib",
        "optax",
        ":config",
        ":mixin",
        "//learning/utilities:logging",
    ],
)
