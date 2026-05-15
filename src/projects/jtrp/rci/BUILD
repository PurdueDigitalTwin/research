load("@jtrp_cuda//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//src/projects/jtrp:__subpackages__"])

# -- libraries -----------------------------------------------------------------
py_library(
    name = "structure",
    srcs = ["structure.py"],
    deps = [requirement("numpy")],
)

py_library(
    name = "calibration",
    srcs = ["calibration.py"],
    deps = [
        requirement("numpy"),
        requirement("opencv-python-headless"),
        requirement("tqdm"),
        ":structure",
    ],
)

py_library(
    name = "georeferencing",
    srcs = ["georeferencing.py"],
    deps = [
        requirement("numpy"),
        requirement("opencv-python-headless"),
        ":structure",
    ],
)

py_library(
    name = "serialization",
    srcs = ["serialization.py"],
    deps = [
        requirement("absl-py"),
        ":structure",
    ],
)

py_library(
    name = "ngsim",
    srcs = ["ngsim.py"],
    deps = [
        requirement("absl-py"),
        requirement("numpy"),
        ":structure",
    ],
)

py_library(
    name = "visualization",
    srcs = ["visualization.py"],
    deps = [
        requirement("absl-py"),
        requirement("opencv-python-headless"),
        requirement("tqdm"),
        ":structure",
    ],
)

py_library(
    name = "gcp_tools",
    srcs = ["gcp_tools.py"],
    deps = [
        requirement("absl-py"),
        requirement("numpy"),
        requirement("opencv-python-headless"),
        ":serialization",
        ":structure",
    ],
)

py_library(
    name = "tracker",
    srcs = ["tracker.py"],
    deps = [
        requirement("absl-py"),
        requirement("lap"),
        requirement("numpy"),
        requirement("opencv-python-headless"),
        requirement("torch"),
        requirement("tqdm"),
        requirement("ultralytics"),
        ":calibration",
        ":georeferencing",
        ":structure",
    ],
)

py_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        requirement("absl-py"),
        ":calibration",
        ":georeferencing",
        ":ngsim",
        ":serialization",
        ":structure",
        ":tracker",
        ":visualization",
    ],
)

# -- single entry point --------------------------------------------------------

py_binary(
    name = "main",
    srcs = ["main.py"],
    data = glob(["constants/*"]),
    deps = [
        requirement("absl-py"),
        requirement("opencv-python-headless"),
        ":calibration",
        ":gcp_tools",
        ":pipeline",
        ":serialization",
    ],
)

# -- tests ---------------------------------------------------------------------

py_test(
    name = "test_structure",
    srcs = ["tests/test_structure.py"],
    deps = [
        requirement("numpy"),
        requirement("pytest"),
        ":structure",
    ],
)

py_test(
    name = "test_calibration",
    srcs = ["tests/test_calibration.py"],
    deps = [
        requirement("numpy"),
        requirement("opencv-python-headless"),
        requirement("pytest"),
        ":calibration",
        ":structure",
    ],
)

py_test(
    name = "test_georeferencing",
    srcs = ["tests/test_georeferencing.py"],
    deps = [
        requirement("numpy"),
        requirement("pytest"),
        ":georeferencing",
        ":structure",
    ],
)

py_test(
    name = "test_ngsim",
    srcs = ["tests/test_ngsim.py"],
    deps = [
        requirement("numpy"),
        requirement("pytest"),
        ":ngsim",
        ":structure",
    ],
)

py_test(
    name = "test_serialization",
    srcs = ["tests/test_serialization.py"],
    deps = [
        requirement("numpy"),
        requirement("pytest"),
        ":georeferencing",
        ":serialization",
        ":structure",
    ],
)

py_test(
    name = "test_visualization",
    srcs = ["tests/test_visualization.py"],
    deps = [
        requirement("numpy"),
        requirement("opencv-python-headless"),
        requirement("pytest"),
        ":structure",
        ":visualization",
    ],
)
