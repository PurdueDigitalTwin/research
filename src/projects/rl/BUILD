load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library")

package(default_visibility = ["//src/projects/rl:__subpackages__"])

ml_py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
        "chex",
        "jax",
        "//src/core:model",
    ],
)

ml_py_library(
    name = "replay_buffer",
    srcs = ["replay_buffer.py"],
    deps = [
        "jax",
        "numpy",
        ":common",
    ],
)

ml_py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        "absl-py",
        "fiddle",
        "jax",
        "//src/core:distributed",
    ],
)
