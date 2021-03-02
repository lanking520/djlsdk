import mxnet as mx
from gluoncv import model_zoo

from mxnet_sdk import export_mxnet


def test_mxnet_gluoncv():
    net = model_zoo.get_model('resnet18_v1', pretrained=True)
    sample_input = [mx.nd.random.uniform(shape=(1, 3, 224, 224))]
    export_mxnet(net, sample_input)


if __name__ == '__main__':
    test_mxnet_gluoncv()
