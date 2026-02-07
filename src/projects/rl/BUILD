load("//third_party:defs.bzl", "ml_py_binary")

package(default_visibility = ["//src/projects/generative:__subpackages__"])

ml_py_binary(
    name = "dqn",
    srcs = ["dqn.py"],
    deps = [
        "flax",
        "gymnasium",
        "jax",
        "matplotlib",
        "optax",
        "//src/core:model",
        "//src/core:train_state",
        "//src/utilities:logging",
    ],
)
