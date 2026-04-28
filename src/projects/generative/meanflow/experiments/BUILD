load("//third_party:defs.bzl", "ml_py_binary")

package(default_visibility = ["//src/projects/generative/meanflow:__subpackages__"])

ml_py_binary(
    name = "run_toy",
    srcs = ["run_toy.py"],
    deps = [
        "absl-py",
        "chex",
        "flax",
        "jax",
        "jaxtyping",
        "numpy",
        "optax",
        "typing_extensions",
        "//src/core:model",
        "//src/core:train_state",
        "//src/projects/generative/meanflow/model:trace",
        "//src/utilities:logging",
    ],
)
