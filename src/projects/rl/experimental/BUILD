load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library")

package(default_visibility = ["//src/projects/rl:__subpackages__"])

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
        "flax",
        "jax",
        "jaxtyping",
        "typing_extensions",
        ":policy",
        "//src/core:model",
        "//src/core:train_state",
        "//src/projects/rl:structure",
        "//src/utilities:logging",
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
        "numpy",
        "optax",
        ":iql",
        "//src/core:trian_state",
        "//src/projects/rl:structure",
        "//src/utilities:logging",
    ],
)
