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
        "//src:all_config",
        "//src/utilities:logging",
    ],
)

ml_py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "fiddle",
        "jax",
        "jaxlib",
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
        "clu",
        "fiddle",
        "flax",
        "jax",
        "jaxlib",
        "optax",
        "orbax-checkpoint",
        ":config",
        ":mixin",
        "//src/utilities:logging",
    ],
)
