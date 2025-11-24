load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "opencv",
    hdrs = glob(["**"]),
    includes = ["."],
    linkopts = [
        "-L/usr/local/lib", 
        "-Wl,-rpath,/usr/local/lib",
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_imgcodecs",
        "-lopencv_dnn",
        "-lopencv_cudaarithm",
        "-lopencv_cudawarping",
    ],
)
