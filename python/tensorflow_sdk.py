import os
import shutil
from typing import Dict

import tensorflow as tf
from base import get_djl_template, zip_dir


def export_tensorflow(model: tf.Module, sample_input: Dict[str, tf.Tensor]):
    directory = "build"
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
        os.mkdir(directory)
    model_path = os.path.join(directory, "tfmodel")
    # Create and stored the model
    tf.saved_model.save(model, model_path + "/")
    zip_dir(model_path, model_path)
    # verify
    saved_model = tf.saved_model.load(model_path + "/")
    print("Current supported signature: " + str(list(saved_model.signatures.keys())))
    print("Try to apply serving_default...")
    infer = saved_model.signatures["serving_default"]
    input_names = []
    for element in infer.structured_input_signature:
        if isinstance(element, Dict):
            input_names += list(element.keys())
    print("Input names: " + str(input_names))
    infer(*sample_input.values())
    # Start creating template
    numpy_dict = {}
    for name, tensor in sample_input.items():
        numpy_dict[name] = tensor_to_numpy(tensor)
    dest = get_djl_template("tfmodel.zip", numpy_dict, add_deps())
    shutil.copy(model_path + ".zip", dest)


def tensor_to_numpy(tensor: tf.Tensor):
    return tensor.numpy()


def add_deps():
    deps = ["dependencies {",
            "  runtimeOnly \"org.apache.logging.log4j:log4j-slf4j-impl:2.12.1\"",
            "  implementation \"ai.djl:api:0.10.0\"",
            "  runtimeOnly \"ai.djl.tensorflow:tensorflow-model-zoo:0.10.0\"",
            "  runtimeOnly \"ai.djl.tensorflow:tensorflow-native-auto:2.3.1\"", "}"]
    return "\n".join(deps)


if __name__ == '__main__':
    tensors = [tf.ones((1, 3, 224)), tf.ones((1, 3, 2))]
    nd = [tensor_to_numpy(x) for x in tensors]
    get_djl_template("model/tensorflow.zip", nd, add_deps())
