load("//third_party:defs.bzl", "ml_py_library", "ml_py_test")

package(default_visibility = ["//src/projects/generative:__subpackages__"])

ml_py_library(
    name = "refinenet",
    srcs = ["refinenet.py"],
    deps = [
        "chex",
        "flax",
        "jax",
    ],
)

ml_py_test(
    name = "test_refinenet",
    srcs = ["test_refinenet.py"],
    deps = [
        "chex",
        "flax",
        "jax",
        ":refinenet",
    ],
)
