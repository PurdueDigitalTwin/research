load("//third_party:defs.bzl", "ml_py_binary")

package(default_visibility = ["//src/projects/generative:__subpackages__"])

ml_py_binary(
    name = "plot_illustration",
    srcs = ["figures/plot_illustration.py"],
    deps = [
        "absl-py",
        "jax",
        "matplotlib",
    ],
)

ml_py_binary(
    name = "plot_toy",
    srcs = ["figures/plot_toy.py"],
    deps = [
        "absl-py",
        "matplotlib",
        "numpy",
    ],
)
