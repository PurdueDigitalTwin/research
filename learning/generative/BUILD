load("//learning:defs.bzl", "ml_py_library", "ml_py_test")

package(default_visibility = ["//learning:__subpackages__"])

ml_py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "fiddle",
        "optax",
        ":meanflow",
        "//learning/core:config",
        "//learning/data:cifar",
        "//learning/data:preprocess",
    ],
)

ml_py_library(
    name = "meanflow",
    srcs = ["meanflow.py"],
    deps = [
        "chex",
        "flax",
        "jax",
        "jaxlib",
        "jaxtyping",
        "typing_extensions",
        "//learning/core:mixin",
        "//learning/generative/model:refinenet",
    ],
)

ml_py_test(
    name = "test_meanflow",
    srcs = ["test_meanflow.py"],
    deps = [
        "chex",
        "flax",
        "jax",
        "jaxlib",
        ":meanflow",
    ],
)
