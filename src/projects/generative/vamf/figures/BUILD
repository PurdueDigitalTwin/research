load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library")

package(default_visibility = ["//src/projects/generative/vamf:__subpackages__"])

ml_py_binary(
    name = "plot_illustration",
    srcs = ["plot_illustration.py"],
    deps = [
        "absl-py",
        "jax",
        "matplotlib",
    ],
)

ml_py_binary(
    name = "plot_toy",
    srcs = ["plot_toy.py"],
    deps = [
        "absl-py",
        "matplotlib",
        "numpy",
        ":_style",
    ],
)

ml_py_binary(
    name = "plot_diagnostics",
    srcs = ["plot_diagnostics.py"],
    deps = [
        "absl-py",
        "matplotlib",
        "numpy",
        ":_style",
    ],
)

ml_py_library(
    name = "_style",
    srcs = ["_style.py"],
    data = [
        "constants/paper.mplstyle",
        "constants/slides.mplstyle",
    ],
    deps = ["matplotlib"],
)

ml_py_binary(
    name = "plot_teaser",
    srcs = ["plot_teaser.py"],
    deps = [
        "absl-py",
        "matplotlib",
        "numpy",
        ":_style",
    ],
)
