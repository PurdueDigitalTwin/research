load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library")

package(default_visibility = ["//src/projects/generative:__subpackages__"])

ml_py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "fiddle",
        "optax",
        ":meanflow",
        "//src/core:config",
        "//src/data:huggingface",
        "//src/data:preprocess",
    ],
)

ml_py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        "absl-py",
        "clu",
        "fiddle",
        "jax",
        "optax",
        "tensorflow",
        ":config",
        "//src/core:config",
        "//src/core:evaluate",
        "//src/core:train",
        "//src/core:train_state",
        "//src/utilities:logging",
    ],
)

ml_py_library(
    name = "meanflow",
    srcs = ["meanflow.py"],
    deps = [
        "chex",
        "flax",
        "jax",
        "jaxtyping",
        "typing_extensions",
        "//src/core:model",
        "//src/core:train_state",
        "//src/projects/generative/model:refinenet",
    ],
)
