import mxnet as mx
from mxnet import gluon

from typing import List

from main import get_djl_template


def export_mxnet(model: gluon.HybridBlock, name: str, sample_input: List[mx.ndarray.NDArray]):
    model.hybridize()
    model.forward(sample_input)
    model.export(name)


def ndarray_to_numpy(nd: mx.ndarray.NDArray):
    return nd.asnumpy()


def add_deps():
    deps = ["dependencies {", "  implementation \"ai.djl:api:0.10.0\"",
            "  runtimeOnly \"ai.djl.mxnet:mxnet-model-zoo:0.10.0\"",
            "  runtimeOnly \"ai.djl.mxnet:mxnet-native-auto:1.7.0-backport\"", "}"]
    return "\n".join(deps)


if __name__ == '__main__':
    mx_nd = [mx.nd.ones((1, 3, 224)), mx.nd.ones((1, 3, 2))]
    nd = [ndarray_to_numpy(x) for x in mx_nd]
    get_djl_template("model/mxnet.zip", nd, add_deps())
