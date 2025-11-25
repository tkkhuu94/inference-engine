load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "detection",
    hdrs = ["detection.h"],
)

cc_library(
    name = "non_max_suppression",
    srcs = ["non_max_suppression.cpp"],
    hdrs = ["non_max_suppression.h"],
    visibility = ["//inference/tests:__subpackages__"],
    deps = [
        ":detection",
        "@opencv",
    ],
)

cc_library(
    name = "inference_params",
    hdrs = ["inference_params.h"],
    deps = ["@opencv"],
)

cc_library(
    name = "inference_engine",
    srcs = ["inference_engine.cpp"],
    hdrs = ["inference_engine.h"],
    visibility = ["//inference/tests:__subpackages__"],
    deps = [
        ":detection",
        ":inference_params",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status:statusor",
        "@opencv",
    ],
)

cc_binary(
    name = "inference",
    srcs = ["inference.cpp"],
    deps = [
        ":inference_engine",
        "@abseil-cpp//absl/log",
        "@opencv",
    ],
)
