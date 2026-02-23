load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library")

package(default_visibility = ["//src/projects/generative:__subpackages__"])

ml_py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        "absl-py",
        "fiddle",
        "jax",
        ":run_ua_flow",
        "//src/core:distributed",
    ],
)

ml_py_library(
    name = "run_ua_flow",
    srcs = ["run_ua_flow.py"],
    deps = [
        "flax",
        "jax",
        "jaxtyping",
        "optax",
        "//src/core:model",
        "//src/core:train_state",
        "//src/projects/generative/model:unet",
    ],
)
