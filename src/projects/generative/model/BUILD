load("//third_party:defs.bzl", "ml_py_library", "ml_py_test")

package(default_visibility = ["//src/projects/generative:__subpackages__"])

ml_py_library(
    name = "dit",
    srcs = ["dit.py"],
    deps = [
        "chex",
        "flax",
        "jax",
        "jaxlib",
    ],
)

ml_py_test(
    name = "test_dit",
    srcs = ["test_dit.py"],
    deps = [
        "flax",
        "jax",
        ":dit",
    ],
)

ml_py_library(
    name = "inception",
    srcs = ["inception.py"],
    deps = [
        "flax",
        "jax",
    ],
)

ml_py_test(
    name = "test_inception",
    srcs = ["test_inception.py"],
    deps = [
        "flax",
        "jax",
        ":inception",
    ],
)

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

ml_py_library(
    name = "unet",
    srcs = ["unet.py"],
    deps = [
        "chex",
        "flax",
        "jax",
    ],
)

ml_py_test(
    name = "test_unet",
    srcs = ["test_unet.py"],
    deps = [
        "jax",
        ":unet",
    ],
)
