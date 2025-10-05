load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load(":defs.bzl", "ml_py_library")

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

# compile pip requirements into a lock file
compile_pip_requirements(
    name = "requirements_cpu",
    timeout = "moderate",
    src = "requirements.in",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_txt = "requirements_lock.txt",
)

compile_pip_requirements(
    name = "requirements_cuda",
    timeout = "moderate",
    src = "requirements_cuda.in",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_txt = "requirements_cuda_lock.txt",
)

compile_pip_requirements(
    name = "requirements_tpu",
    timeout = "moderate",
    src = "requirements_tpu.in",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_txt = "requirements_tpu_lock.txt",
)

# add all configuration
ml_py_library(
    name = "all_config",
    deps = [
        "//learning/generative:config",
        # add more config targets here
    ],
)
