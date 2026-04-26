load("//third_party:defs.bzl", "ml_py_binary")

package(default_visibility = ["//src/projects/generative:__subpackages__"])

ml_py_binary(
    name = "run_toy",
    srcs = ["experiments/run_toy.py"],
    deps = [
        "absl-py",
        "chex",
        "flax",
        "jax",
        "jaxtyping",
        "optax",
        "typing_extensions",
        "//src/core:model",
        "//src/core:train_state",
        "//src/utilities:logging",
    ],
)

ml_py_binary(
    name = "plot_illustration",
    srcs = ["figures/plot_illustration.py"],
    deps = [
        "absl-py",
        "jax",
        "matplotlib",
    ],
)
