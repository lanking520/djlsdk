import torch
import torchvision as torchvision

from pytorch_sdk import export_pytorch


def test_pytorch_torchvision():
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    sample_input = [torch.rand(1, 3, 224, 224)]
    export_pytorch(model, sample_input)


if __name__ == '__main__':
    test_pytorch_torchvision()
