from typing import List

import tensorflow as tf
from base import get_djl_template


def export_tensorflow(model: tf.Module, name: str, sample_input: List[tf.Tensor]):
    tf.saved_model.save(model, name + "/1/")


def tensor_to_numpy(tensor: tf.Tensor):
    return tensor.numpy()


def add_deps():
    deps = ["dependencies {", "  implementation \"ai.djl:api:0.10.0\"",
            "  runtimeOnly \"ai.djl.tensorflow:tensorflow-model-zoo:0.10.0\"",
            "  runtimeOnly \"ai.djl.tensorflow:tensorflow-native-auto:2.4.1\"", "}"]
    return "\n".join(deps)


if __name__ == '__main__':
    tensors = [tf.ones((1, 3, 224)), tf.ones((1, 3, 2))]
    nd = [tensor_to_numpy(x) for x in tensors]
    get_djl_template("model/tensorflow.zip", nd, add_deps())
