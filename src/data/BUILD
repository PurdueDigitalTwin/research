load("//third_party:defs.bzl", "ml_py_library", "ml_py_test")

package(default_visibility = ["//src:__subpackages__"])

ml_py_library(
    name = "huggingface",
    srcs = ["huggingface.py"],
    deps = [
        "Pillow",
        "datasets",
        "jax",
        "jaxtyping",
        "tensorflow",
        "typing_extensions",
        "//src/core:datamodule",
    ],
)

ml_py_test(
    name = "test_huggingface",
    srcs = ["test_huggingface.py"],
    deps = [
        "jax",
        "numpy",
        "tensorflow",
        ":huggingface",
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
