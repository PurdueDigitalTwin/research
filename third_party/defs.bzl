"""Customized build rules for learning module."""

load("@ml_infra_cpu_3_10//:requirements.bzl", cpu_req = "requirement")
load("@ml_infra_cuda_3_10//:requirements.bzl", cuda_req = "requirement")
load("@ml_infra_mps_3_10//:requirements.bzl", mps_req = "requirement")
load("@ml_infra_tpu_3_10//:requirements.bzl", tpu_req = "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

def _select_requirement(name):
    """Select from dependency based on ML platform.

    Args:
        name (str): name of the target to depend on.

    Returns:
        A select statement that picks the right dependency based on the
    """
    if name in ["tensorflow-cpu"]:
        return select({
            "//third_party:is_cpu": [cpu_req(name)],
            "//third_party:is_cuda": [cuda_req(name)],
            "//third_party:is_mps": [mps_req("tensorflow")],
            "//third_party:is_tpu": [tpu_req(name)],
        })

    return select({
        "//third_party:is_cpu": [cpu_req(name)],
        "//third_party:is_cuda": [cuda_req(name)],
        "//third_party:is_mps": [mps_req(name)],
        "//third_party:is_tpu": [tpu_req(name)],
    })

def _select_all_requirements(names = []):
    """Returns a compiled list of all requirements selected by ML platform.

    Args:
        names (list): list of target names to depend on.

    Returns:
        A list of dependencies based on the ML platform.
    """
    reqs = []
    for name in names:
        reqs += _select_requirement(name)

    if "fiddle" in names and "etils" not in names:
        reqs += _select_requirement("etils")

    if "jax" in names:
        if "jaxlib" not in names:
            reqs += _select_requirement("jaxlib")
        reqs += select({
            "//third_party:is_mps": [mps_req("jax-metal")],
            "//conditions:default": [],
        })

    return reqs

def _partition_deps(deps = []):
    """Partitions dependencies into native and other dependencies.

    Args:
        deps (list): list of dependencies to partition.

    Returns:
        A tuple of (native_deps, other_deps).
    """
    native_deps = []
    other_deps = []
    for dep in deps:
        if ":" in dep or dep.startswith("//"):
            native_deps.append(dep)
        else:
            other_deps.append(dep)

    return native_deps, other_deps

def ml_py_binary(name, **kwargs):
    """Creates a Python binary with all dependencies included."""
    native_deps, other_deps = _partition_deps(kwargs.pop("deps", []))
    kwargs["deps"] = native_deps + _select_all_requirements(other_deps)

    return py_binary(
        name = name,
        **kwargs
    )

def ml_py_library(name, **kwargs):
    """Creates a Python library with all dependencies included."""
    native_deps, other_deps = _partition_deps(kwargs.pop("deps", []))
    kwargs["deps"] = native_deps + _select_all_requirements(other_deps)

    return py_library(
        name = name,
        **kwargs
    )

def ml_py_test(name, **kwargs):
    """Creates a Python test with all dependencies included."""
    native_deps, other_deps = _partition_deps(kwargs.pop("deps", []))
    kwargs["deps"] = (
        native_deps +
        _select_all_requirements(other_deps) +
        _select_requirement("pytest") +
        _select_requirement("pytest-cov")
    )

    return py_test(
        name = name,
        **kwargs
    )
