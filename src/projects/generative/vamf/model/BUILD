load("//third_party:defs.bzl", "ml_py_library", "ml_py_test")

package(default_visibility = ["//src/projects/generative/vamf:__subpackages__"])

ml_py_library(
    name = "trace",
    srcs = ["trace.py"],
    deps = ["jax"],
)

ml_py_test(
    name = "test_trace",
    srcs = ["tests/test_trace.py"],
    deps = [
        "jax",
        ":trace",
    ],
)
