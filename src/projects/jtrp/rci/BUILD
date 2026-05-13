load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library", "ml_py_test")

package(default_visibility = ["//src/projects/jtrp:__subpackages__"])

# -- libraries -----------------------------------------------------------------

ml_py_library(
    name = "structure",
    srcs = ["structure.py"],
    deps = ["numpy"],
)

ml_py_library(
    name = "calibration",
    srcs = ["calibration.py"],
    deps = [
        "numpy",
        "opencv-python-headless",
        ":structure",
    ],
)

ml_py_library(
    name = "georeferencing",
    srcs = ["georeferencing.py"],
    deps = [
        "numpy",
        "opencv-python-headless",
        ":structure",
    ],
)

ml_py_library(
    name = "serialization",
    srcs = ["serialization.py"],
    deps = [
        ":structure",
        "//src/utilities:logging",
    ],
)

ml_py_library(
    name = "ngsim",
    srcs = ["ngsim.py"],
    deps = [
        "numpy",
        ":structure",
        "//src/utilities:logging",
    ],
)

ml_py_library(
    name = "visualization",
    srcs = ["visualization.py"],
    deps = [
        "opencv-python-headless",
        "tqdm",
        ":structure",
        "//src/utilities:logging",
    ],
)

ml_py_library(
    name = "gcp_tools",
    srcs = ["gcp_tools.py"],
    deps = [
        "numpy",
        "opencv-python-headless",
        ":serialization",
        ":structure",
        "//src/utilities:logging",
    ],
)

ml_py_library(
    name = "tracker",
    srcs = ["tracker.py"],
    deps = [
        "lap",
        "numpy",
        "opencv-python-headless",
        "torch",
        "tqdm",
        "ultralytics",
        ":calibration",
        ":georeferencing",
        ":structure",
        "//src/utilities:logging",
    ],
)

ml_py_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        ":calibration",
        ":georeferencing",
        ":ngsim",
        ":serialization",
        ":structure",
        ":tracker",
        ":visualization",
        "//src/utilities:logging",
    ],
)

# -- single entry point --------------------------------------------------------

ml_py_binary(
    name = "main",
    srcs = ["main.py"],
    data = glob(["constants/*"]),
    deps = [
        "absl-py",
        ":calibration",
        ":gcp_tools",
        ":pipeline",
        ":serialization",
        "//src/utilities:logging",
    ],
)

# -- tests ---------------------------------------------------------------------

ml_py_test(
    name = "test_structure",
    srcs = ["tests/test_structure.py"],
    deps = [
        "numpy",
        ":structure",
    ],
)

ml_py_test(
    name = "test_calibration",
    srcs = ["tests/test_calibration.py"],
    deps = [
        "numpy",
        "opencv-python-headless",
        ":calibration",
        ":structure",
    ],
)

ml_py_test(
    name = "test_georeferencing",
    srcs = ["tests/test_georeferencing.py"],
    deps = [
        "numpy",
        ":georeferencing",
        ":structure",
    ],
)

ml_py_test(
    name = "test_ngsim",
    srcs = ["tests/test_ngsim.py"],
    deps = [
        "numpy",
        ":ngsim",
        ":structure",
    ],
)

ml_py_test(
    name = "test_serialization",
    srcs = ["tests/test_serialization.py"],
    deps = [
        "numpy",
        ":georeferencing",
        ":serialization",
        ":structure",
    ],
)

ml_py_test(
    name = "test_visualization",
    srcs = ["tests/test_visualization.py"],
    deps = [
        "numpy",
        "opencv-python-headless",
        ":structure",
        ":visualization",
    ],
)
