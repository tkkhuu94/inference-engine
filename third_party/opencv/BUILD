load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "opencv",
    hdrs = glob(["**"]),
    includes = ["."],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_imgcodecs",
        "-lopencv_dnn",
    ],
)
