load("//third_party:defs.bzl", "ml_py_binary")

package(default_visibility = ["//src/projects/rl:__subpackages__"])

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
        "//src/core:model",
        "//src/core:train_state",
        "//src/projects/rl:replay_buffer",
        "//src/projects/rl:structure",
        "//src/utilities:logging",
    ],
)
