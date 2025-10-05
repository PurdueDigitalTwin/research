load("//learning:defs.bzl", "ml_py_library", "ml_py_test")

package(default_visibility = ["//learning/generative:__subpackages__"])

ml_py_library(
    name = "refinenet",
    srcs = ["refinenet.py"],
    deps = [
        "chex",
        "flax",
        "jax",
        "jaxlib",
    ],
)

ml_py_test(
    name = "test_refinenet",
    srcs = ["test_refinenet.py"],
    deps = [
        "chex",
        "flax",
        "jax",
        "jaxlib",
        ":refinenet",
    ],
)
