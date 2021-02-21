import numpy
import os
import shutil
from typing import List, Tuple

MAGIC_NUMBER = "NDAR"
VERSION = 2


def get_djl_template(model_path: str, sample_input: List[numpy.ndarray], deps: str):
    lines = ["# DJL example",
             "Install Java 8 or above"
             "To start with, just run the followings:\n",
             "```",
             "./gradlew run --args=\"" + model_path + "\"",
             "```"]
    directory = "build/template"
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
    shutil.copytree("templates/gradle_project", "build/template")
    with open(directory + "/README.md", "w") as f:
        f.write("\n".join(lines))
    with open(directory + "/build.gradle", "a") as f:
        f.write(deps)
    with open(directory + "/src/main/java/ai/djl/examples/Example.java", "a") as f:
        for nd in sample_input:
            shape = nd.shape
            dtype = str(nd.dtype).upper()
            f.write("    list.add(manager.create(new Shape" + str(shape) + ", DataType." + dtype + "));\n")
        f.write("    predictor.predict(list);\n  }\n}")


def djl_encode(ndlist: List[numpy.ndarray]) -> bytearray:
    arr = bytearray()
    arr.append(len(ndlist))
    for nd in ndlist:
        arr.extend(bytes(MAGIC_NUMBER, "utf8"))
        arr.append(VERSION)
        arr.append(0)
        arr.extend(bytes(str(nd.dtype).upper(), "utf8"))
        shape_encode(nd.shape, arr)
        arr.extend(nd.newbyteorder('>').tobytes("C"))  # make it big endian
    return arr


def shape_encode(shape: Tuple[int], arr: bytearray):
    arr.append(len(shape))
    arr += bytearray(shape)
    for _ in shape:
        arr.extend(bytes("?", "utf8"))


def djl_decode(encoded: bytearray) -> List[numpy.ndarray]:
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ndlist = [numpy.ones((1, 3, 2))]
    encoded = djl_encode(ndlist)
    djl_decode(encoded)
