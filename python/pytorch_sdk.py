from typing import List
import torch

from base import get_djl_template


def export_pytorch(module: torch.nn.Module, name: str, sample_input: List[torch.Tensor]):
    traced_module = torch.jit.trace(module, tuple(sample_input))
    traced_module.save(name + ".pt")


def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.numpy()


def add_deps():
    deps = ["dependencies {", "  implementation \"ai.djl:api:0.10.0\"",
            "  runtimeOnly \"ai.djl.pytorch:pytorch-model-zoo:0.10.0\"",
            "  runtimeOnly \"ai.djl.pytorch:pytorch-native-auto:1.7.1\"", "}"]
    return "\n".join(deps)


if __name__ == '__main__':
    tensors = [torch.ones((1, 3, 224)), torch.ones((1, 3, 2))]
    nd = [tensor_to_numpy(x) for x in tensors]
    get_djl_template("model/pytorch.zip", nd, add_deps())
