load("//third_party:defs.bzl", "ml_py_library", "ml_py_test")

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
        "//src/core:model",
        "//src/core:train_state",
        "//src/projects/rl:common",
        "//src/utilities:logging",
    ],
)

ml_py_test(
    name = "test_dqn",
    srcs = ["test_dqn.py"],
    deps = [
        "flax",
        "jax",
        "optax",
        ":dqn",
        "//src/projects/rl:common",
    ],
)
