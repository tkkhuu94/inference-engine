load("@rules_cc//cc:defs.bzl", "cc_test")

cc_test(
    name = "test_inference_engine",
    srcs = ["test_inference_engine.cpp"],
    deps = [
        "//inference:inference_engine",
        "@googletest//:gtest_main",
        "@opencv",
    ],
)
