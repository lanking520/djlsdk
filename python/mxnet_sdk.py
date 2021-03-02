import os
import shutil
from typing import List

import mxnet as mx
from mxnet import gluon

from base import get_djl_template, zip_files


def export_mxnet(model: gluon.HybridBlock, sample_input: List[mx.ndarray.NDArray]):
    model.hybridize()
    model(*sample_input)
    directory = "build"
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
        os.mkdir(directory)
    model_path = os.path.join(directory, "mxmodel")
    # Create and stored the model
    model.export(model_path)
    zip_files(directory, model_path)
    # Start creating template
    array = [ndarray_to_numpy(x) for x in sample_input]
    dest = get_djl_template("mxmodel.zip", array, add_deps())
    shutil.copy(model_path + ".zip", dest)


def ndarray_to_numpy(nd: mx.ndarray.NDArray):
    return nd.asnumpy()


def add_deps():
    deps = ["dependencies {",
            "  runtimeOnly \"org.apache.logging.log4j:log4j-slf4j-impl:2.12.1\"",
            "  implementation \"ai.djl:api:0.10.0\"",
            "  runtimeOnly \"ai.djl.mxnet:mxnet-model-zoo:0.10.0\"",
            "  runtimeOnly \"ai.djl.mxnet:mxnet-native-auto:1.7.0-backport\"", "}"]
    return "\n".join(deps)


if __name__ == '__main__':
    mx_nd = [mx.nd.ones((1, 3, 224)), mx.nd.ones((1, 3, 2))]
    nd = [ndarray_to_numpy(x) for x in mx_nd]
    get_djl_template("model/mxnet.zip", nd, add_deps())
