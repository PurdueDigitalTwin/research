load("//third_party:defs.bzl", "ml_py_library")

package(default_visibility = ["//src:__subpackages__"])

ml_py_library(
    name = "logging",
    srcs = ["logging.py"],
    deps = [
        "absl-py",
        "jax",
    ],
)

ml_py_library(
    name = "rank_zero",
    srcs = ["rank_zero.py"],
    deps = [
        "jax",
    ],
)

ml_py_library(
    name = "training",
    srcs = ["training.py"],
    deps = [
        "jax",
        "jaxtyping",
    ],
)

ml_py_library(
    name = "visualization",
    srcs = ["visualization.py"],
    deps = [
        "jax",
        ":logging",
    ],
)
