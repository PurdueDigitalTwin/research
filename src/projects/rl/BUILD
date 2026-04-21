load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library")

package(default_visibility = ["//src/projects/rl:__subpackages__"])

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
    name = "iql",
    srcs = ["iql.py"],
    deps = [
        "chex",
        "flax",
        "jax",
        "jaxtyping",
        "typing_extensions",
        ":policy",
        ":structure",
        "//src/core:model",
        "//src/core:train_state",
    ],
)

ml_py_library(
    name = "ppo",
    srcs = ["ppo.py"],
    deps = [
        "flax",
        "jax",
        "jaxtyping",
        "optax",
        "typing_extensions",
        ":policy",
        "//src/core:model",
    ],
)

ml_py_library(
    name = "ppo_continuous",
    srcs = ["ppo_continuous.py"],
    deps = [
        "flax",
        "jax",
        "jaxtyping",
        "optax",
        "typing_extensions",
        ":policy",
        "//src/core:model",
    ],
)

ml_py_library(
    name = "replay_buffer",
    srcs = ["replay_buffer.py"],
    deps = [
        "jax",
        "numpy",
        ":structure",
    ],
)

ml_py_library(
    name = "structure",
    srcs = ["structure.py"],
    deps = [
        "chex",
        "jax",
    ],
)

ml_py_binary(
    name = "run_dqn",
    srcs = ["run_dqn.py"],
    deps = [
        "absl-py",
        "flax",
        "gymnasium",
        "jax",
        "matplotlib",
        "optax",
        ":dqn",
        ":replay_buffer",
        ":structure",
        "//src/core:train_state",
        "//src/utilities:logging",
        "//src/utilities:training",
    ],
)

ml_py_binary(
    name = "run_iql",
    srcs = ["run_iql.py"],
    deps = [
        "absl-py",
        "gymnasium",
        "jax",
        "matplotlib",
        "minari",
        "mujoco",
        "numpy",
        "optax",
        ":iql",
        ":structure",
        "//src/core:train_state",
        "//src/utilities:logging",
    ],
)

ml_py_binary(
    name = "run_ac",
    srcs = ["run_ac.py"],
    deps = [
        "absl-py",
        "flax",
        "gymnasium",
        "jax",
        "jaxtyping",
        "matplotlib",
        "optax",
        ":replay_buffer",
        ":structure",
        "//src/core:model",
        "//src/core:train_state",
        "//src/utilities:logging",
    ],
)

ml_py_binary(
    name = "run_ppo_cartpole",
    srcs = ["run_ppo_cartpole.py"],
    deps = [
        "absl-py",
        "flax",
        "gymnasium",
        "jax",
        "jaxtyping",
        "matplotlib",
        "numpy",
        "optax",
        ":ppo",
        "//src/core:model",
        "//src/core:train_state",
        "//src/utilities:logging",
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
        ":ppo",
        "//src/core:model",
        "//src/core:train_state",
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
