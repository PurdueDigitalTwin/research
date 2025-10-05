load("//learning:defs.bzl", "ml_py_library", "ml_py_test")

package(default_visibility = ["//learning:__subpackages__"])

ml_py_library(
    name = "datamodule",
    srcs = ["datamodule.py"],
    deps = [
        "jax",
        "jaxlib",
        "numpy",
        "tensorflow",
        "tensorflow_datasets",
        "//learning/core:mixin",
        "//learning/utilities:logging",
    ],
)

ml_py_library(
    name = "cifar",
    srcs = ["cifar.py"],
    deps = [
        "jax",
        "jaxlib",
        "numpy",
        "tensorflow_datasets",
        ":datamodule",
    ],
)

ml_py_test(
    name = "test_cifar",
    srcs = ["test_cifar.py"],
    deps = [
        "chex",
        "jax",
        "jaxlib",
        ":cifar",
        "//learning/core:mixin",
    ],
)

ml_py_library(
    name = "mnist",
    srcs = ["mnist.py"],
    deps = [
        "jax",
        "jaxlib",
        "numpy",
        "tensorflow_datasets",
        ":datamodule",
    ],
)

ml_py_test(
    name = "test_mnist",
    srcs = ["test_mnist.py"],
    deps = [
        "chex",
        "jax",
        "jaxlib",
        ":mnist",
        "//learning/core:mixin",
    ],
)

ml_py_library(
    name = "preprocess",
    srcs = ["preprocess.py"],
    deps = [
        "tensorflow",
    ],
)

ml_py_test(
    name = "test_preprocess",
    srcs = ["test_preprocess.py"],
    deps = [
        "numpy",
        "tensorflow",
        ":preprocess",
    ],
)
