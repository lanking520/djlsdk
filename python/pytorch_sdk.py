import os
import shutil
from typing import List
import torch

from base import get_djl_template, zip_files


def export_pytorch(module: torch.nn.Module, sample_input: List[torch.Tensor]):
    traced_module = torch.jit.trace(module, tuple(sample_input))
    directory = "build"
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
        os.mkdir(directory)
    model_path = os.path.join(directory, "ptmodel")
    traced_module.save(model_path + ".pt")
    zip_files(directory, model_path)
    # Start creating template
    array = [tensor_to_numpy(x) for x in sample_input]
    dest = get_djl_template("ptmodel.zip", array, add_deps())
    shutil.copy(model_path + ".zip", dest)


def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.numpy()


def add_deps():
    deps = ["dependencies {",
            "  runtimeOnly \"org.apache.logging.log4j:log4j-slf4j-impl:2.12.1\"",
            "  implementation \"ai.djl:api:0.10.0\"",
            "  runtimeOnly \"ai.djl.pytorch:pytorch-model-zoo:0.10.0\"",
            "  runtimeOnly \"ai.djl.pytorch:pytorch-native-auto:1.7.1\"", "}"]
    return "\n".join(deps)


if __name__ == '__main__':
    tensors = [torch.ones((1, 3, 224)), torch.ones((1, 3, 2))]
    nd = [tensor_to_numpy(x) for x in tensors]
    get_djl_template("model/pytorch.zip", nd, add_deps())
