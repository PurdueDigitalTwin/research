load("//third_party:defs.bzl", "ml_py_binary")

package(default_visibility = ["//src/projects/generative/vamf:__subpackages__"])

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
        "//src/projects/generative/vamf/model:beta_schedule",
        "//src/projects/generative/vamf/model:trace",
        "//src/utilities:logging",
    ],
)

ml_py_binary(
    name = "run_diagnostic_toy",
    srcs = ["run_diagnostic_toy.py"],
    deps = [
        "absl-py",
        "flax",
        "jax",
        "numpy",
        "optax",
        "//src/core:train_state",
        "//src/projects/generative/vamf/model:diagnostic",
    ],
)

ml_py_binary(
    name = "run_diagnostic_unet",
    srcs = ["run_diagnostic_unet.py"],
    deps = [
        "Pillow",
        "datasets",
        "etils",
        "jax",
        "jaxtyping",
        "numpy",
        "orbax-checkpoint",
        "//src/projects/generative:meanflow",
        "//src/projects/generative/vamf/model:diagnostic",
    ],
)

ml_py_binary(
    name = "probe_dit_checkpoint",
    srcs = ["probe_dit_checkpoint.py"],
    deps = [
        "absl-py",
        "etils",
        "fiddle",
        "jax",
        "numpy",
        "orbax-checkpoint",
        "//src/projects/generative:config",
        "//src/projects/generative/vamf/model:diagnostic",
        "//src/utilities:logging",
    ],
)

ml_py_binary(
    name = "sample_dit_qualitative",
    srcs = ["sample_dit_qualitative.py"],
    deps = [
        "absl-py",
        "etils",
        "fiddle",
        "flax",
        "jax",
        "jaxtyping",
        "matplotlib",
        "numpy",
        "orbax-checkpoint",
        "//src/projects/generative:config",
        "//src/utilities:logging",
    ],
)
