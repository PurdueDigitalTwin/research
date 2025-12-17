load("//third_party:defs.bzl", "ml_py_library", "ml_py_test")

package(default_visibility = ["//src/projects/generative:__subpackages__"])

ml_py_library(
    name = "augment",
    srcs = ["augment.py"],
    deps = [
        "chex",
        "flax",
        "jax",
    ],
)

ml_py_test(
    name = "test_augment",
    srcs = ["test_augment.py"],
    deps = [
        "chex",
        "jax",
        ":augment",
    ],
)
