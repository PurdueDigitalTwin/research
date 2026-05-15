load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(default_visibility = ["//visibility:public"])

compile_pip_requirements(
    name = "requirements_cpu",
    timeout = "moderate",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_in = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)

compile_pip_requirements(
    name = "requirements_cuda",
    timeout = "moderate",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_in = "requirements_cuda.in",
    requirements_txt = "requirements_cuda_lock.txt",
)
