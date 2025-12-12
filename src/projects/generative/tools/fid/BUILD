load("//third_party:defs.bzl", "ml_py_library", "ml_py_test")

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
