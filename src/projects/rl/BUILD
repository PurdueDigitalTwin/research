load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library")

package(default_visibility = ["//src/projects/generative:__subpackages__"])

ml_py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
        "chex",
        "jax",
    ],
)

ml_py_library(
    name = "dqn",
    srcs = ["dqn.py"],
    deps = [
        "flax",
        "gymnasium",
        "jax",
        "jaxtyping",
        "optax",
        "typing_extensions",
        ":policy",
        "//src/core:model",
        "//src/core:train_state",
        "//src/utilities:logging",
    ],
)

ml_py_library(
    name = "policy",
    srcs = ["policy.py"],
    deps = [
        "flax",
        "jax",
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
