load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library", "ml_py_test")

package(default_visibility = ["//src/projects/generative:__subpackages__"])

ml_py_library(
    name = "fid",
    srcs = ["fid.py"],
    deps = [
        "Pillow",
        "chex",
        "datasets",
        "flax",
        "huggingface_hub",
        "jax",
        "jaxtyping",
        "numpy",
        "scipy",
        "tqdm",
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

ml_py_binary(
    name = "diagnostic",
    srcs = ["diagnostic.py"],
    deps = [
        "Pillow",
        "datasets",
        "etils",
        "jax",
        "jaxtyping",
        "numpy",
        "orbax-checkpoint",
        "//src/projects/generative:meanflow",
    ],
)

ml_py_binary(
    name = "encode_latents",
    srcs = ["encode_latents.py"],
    deps = [
        "absl-py",
        "jax",
        "numpy",
        "tensorflow-cpu",
        "//src/data:huggingface",
        "//src/data:preprocess",
        "//src/projects/generative/model:vae",
    ],
)
