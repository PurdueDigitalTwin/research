load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library")

package(default_visibility = ["//src/projects/rl:__subpackages__"])

ml_py_library(
    name = "ppo_continuous",
    srcs = ["ppo_continuous.py"],
    deps = [
        "flax",
        "jax",
        "jaxtyping",
        "optax",
        "typing_extensions",
        "//src/core:model",
        "//src/projects/rl:policy",
    ],
)

ml_py_binary(
    name = "run_ppo_mountaincar",
    srcs = ["run_ppo_mountaincar.py"],
    deps = [
        "absl-py",
        "flax",
        "gymnasium",
        "jax",
        "jaxtyping",
        "matplotlib",
        "numpy",
        "optax",
        "//src/core:model",
        "//src/core:train_state",
        "//src/projects/rl:ppo",
        "//src/utilities:logging",
    ],
)

ml_py_binary(
    name = "run_ppo_pendulum",
    srcs = ["run_ppo_pendulum.py"],
    deps = [
        "absl-py",
        "flax",
        "gymnasium",
        "jax",
        "jaxtyping",
        "matplotlib",
        "numpy",
        "optax",
        ":ppo_continuous",
        "//src/core:model",
        "//src/core:train_state",
        "//src/utilities:logging",
    ],
)
