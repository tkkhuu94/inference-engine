load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "image_info",
    hdrs = ["image_info.h"],
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
        ":inference_params",
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
    ],
)
