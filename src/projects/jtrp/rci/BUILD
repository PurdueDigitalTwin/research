load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library", "ml_py_test")

package(default_visibility = ["//src/projects/jtrp:__subpackages__"])

ml_py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        "absl-py",
        "//src/utilities:logging",
    ],
)

ml_py_library(
    name = "structure",
    srcs = ["structure.py"],
    deps = ["numpy"],
)

ml_py_test(
    name = "test_structure",
    srcs = ["tests/test_structure.py"],
    deps = [
        "numpy",
        ":structure",
    ],
)

ml_py_library(
    name = "calibration",
    srcs = ["calibration.py"],
    deps = [
        "numpy",
        "opencv-python-headless",
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
