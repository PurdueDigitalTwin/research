load("//third_party:defs.bzl", "ml_py_test")

package(default_visibility = ["//src/projects/rl:__subpackages__"])

ml_py_test(
    name = "test_policy",
    srcs = ["test_policy.py"],
    deps = [
        "flax",
        "jax",
        "//src/projects/rl:policy",
    ],
)

ml_py_test(
    name = "test_iql",
    srcs = ["test_iql.py"],
    deps = [
        "jax",
        "optax",
        "//src/core:model",
        "//src/core:train_state",
        "//src/projects/rl:iql",
        "//src/projects/rl:structure",
    ],
)
