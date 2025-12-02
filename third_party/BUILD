load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(default_visibility = ["//visibility:public"])

# configuration settings for different ML platforms
config_setting(
    name = "is_cpu",
    define_values = {"ml_platform": "cpu"},
)

config_setting(
    name = "is_cuda",
    define_values = {"ml_platform": "cuda"},
)

config_setting(
    name = "is_tpu",
    define_values = {"ml_platform": "tpu"},
)

config_setting(
    name = "is_mps",
    define_values = {"ml_platform": "mps"},
)

# compile pip requirements into a lock file
compile_pip_requirements(
    name = "requirements_3_10_cpu",
    timeout = "moderate",
    srcs = ["requirements.in"],
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_txt = "requirements_3_10_lock.txt",
)

compile_pip_requirements(
    name = "requirements_3_10_cuda",
    timeout = "moderate",
    srcs = [
        "requirements.in",
        "requirements_cuda.in",
    ],
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_txt = "requirements_3_10_cuda_lock.txt",
)

compile_pip_requirements(
    name = "requirements_3_10_tpu",
    timeout = "moderate",
    srcs = [
        "requirements.in",
        "requirements_tpu.in",
    ],
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_txt = "requirements_3_10_tpu_lock.txt",
)

compile_pip_requirements(
    name = "requirements_3_10_mps",
    timeout = "moderate",
    srcs = [
        "requirements.in",
        "requirements_mps.in",
    ],
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_txt = "requirements_3_10_mps_lock.txt",
)
