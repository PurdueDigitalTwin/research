load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library", "ml_py_test")

package(default_visibility = ["//src/projects/jtrp:__subpackages__"])

ml_py_library(
    name = "structure",
    srcs = ["structure.py"],
    deps = [
        "numpy",
    ],
)

ml_py_library(
    name = "tracker",
    srcs = ["tracker.py"],
    deps = [
        "absl-py",
        "lap",
        "numpy",
        "opencv-python-headless",
        "tqdm",
        "ultralytics",
        ":structure",
    ],
)

ml_py_library(
    name = "serialization",
    srcs = ["serialization.py"],
    deps = [
        "absl-py",
        ":structure",
    ],
)

ml_py_library(
    name = "visualization",
    srcs = ["visualization.py"],
    deps = [
        "absl-py",
        "numpy",
        "opencv-python-headless",
        ":structure",
        "//src/utilities:logging",
    ],
)

ml_py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        "absl-py",
        ":serialization",
        ":tracker",
        ":visualization",
        "//src/utilities:logging",
    ],
)

ml_py_test(
    name = "test_structure",
    srcs = ["test_structure.py"],
    deps = [
        "numpy",
        ":structure",
    ],
)

ml_py_test(
    name = "test_serialization",
    srcs = ["test_serialization.py"],
    deps = [
        "numpy",
        ":serialization",
        ":structure",
        "//src/utilities:logging",
    ],
)

ml_py_test(
    name = "test_visualization",
    srcs = ["test_visualization.py"],
    deps = [
        "numpy",
        "opencv-python-headless",
        ":structure",
        ":visualization",
    ],
)
