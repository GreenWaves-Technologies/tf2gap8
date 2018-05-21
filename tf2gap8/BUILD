#third_party/tf2gap8/BUILD

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

cc_library(
    name = "node",
    srcs = ["node.cc"],
    hdrs = ["node.h"],
    deps = [
    	 ":colormod",
    	 "//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "Layer",
    srcs = ["Layer.cc"],
    hdrs = ["Layer.h"],
)

cc_library(
    name = "GAP8Tensor",
    srcs = ["GAP8Tensor.cc"],
    hdrs = ["GAP8Tensor.h"],
    deps = [
        "//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "ConvLayer",
    srcs = ["ConvLayer.cc"],
    hdrs = ["ConvLayer.h"],
    deps = [":Layer"],
)


cc_library(
    name = "PoolLayer",
    srcs = ["PoolLayer.cc"],
    hdrs = ["PoolLayer.h"],
    deps = [":Layer"],
)

cc_library(
    name = "DenseLayer",
    srcs = ["DenseLayer.cc"],
    hdrs = ["DenseLayer.h"],
    deps = [":Layer"],
)

cc_library(
    name= "tf2gap8-exception",
    hdrs = ["tf2gap8-exception.h"],
)

cc_library(
    name = "colormod",
    hdrs = ["colormod.h"],
)
    
tf_cc_binary(
    name = "tf2gap8",
    srcs = ["tf2gap8.cc"],
    deps = [
        "//tensorflow/core:tensorflow",
        ":node",
        ":Layer",
	":ConvLayer",
        ":PoolLayer",
        ":DenseLayer",
        ":GAP8Tensor",
	":tf2gap8-exception",
	":colormod",

    ],
)

