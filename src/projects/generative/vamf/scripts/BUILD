load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

package(default_visibility = ["//src/projects/generative/vamf:__subpackages__"])

sh_binary(
    name = "run_toy_experiment",
    srcs = ["run_toy_experiment.sh"],
    data = ["//src/projects/generative/vamf/experiments:run_toy"],
    deps = ["@bazel_tools//tools/bash/runfiles"],
)
