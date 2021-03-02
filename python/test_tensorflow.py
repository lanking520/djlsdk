import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_sdk import export_tensorflow


def test_tensorflow_keras():
    model = keras.applications.MobileNetV2()
    sample_input = {"input_1": tf.random.uniform((1, 224, 224, 3))}
    export_tensorflow(model, sample_input)


if __name__ == '__main__':
    test_tensorflow_keras()
