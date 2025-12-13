load("//third_party:defs.bzl", "ml_py_library", "ml_py_test")

package(default_visibility = ["//src/projects/generative:__subpackages__"])

ml_py_library(
    name = "fid",
    srcs = ["fid.py"],
    deps = [
        "chex",
        "jax",
        "numpy",
        "scipy",
        "//src/projects/generative/model:inception",
        "//src/utilities:logging",
    ],
)

ml_py_test(
    name = "test_fid",
    srcs = ["test_fid.py"],
    deps = [
        "numpy",
        ":fid",
    ],
)
