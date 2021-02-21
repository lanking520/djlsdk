import numpy
import os
import shutil
from typing import List


def get_djl_template(model_path: str, sample_input: List[numpy.ndarray], deps: str):
    lines = ["# DJL example",
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
