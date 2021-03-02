import numpy as np
import os
import shutil
import zipfile
from typing import List, Dict


def get_djl_template(model_path: str, sample_input: List[np.ndarray] or Dict[str, np.array], deps: str) -> str:
    lines = ["# DJL example",
             "Install Java 8 or above.",
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
        write_input(f, sample_input)
    return directory


def write_input(f, sample_input):
    if isinstance(sample_input, Dict):
        for name, nd in sample_input.items():
            shape = nd.shape
            dtype = str(nd.dtype).upper()
            f.write("    NDArray " + name + " = manager.ones(new Shape" + str(shape) + ", DataType." + dtype + ");\n")
            f.write("    " + name + ".setName(\"" + name + "\");\n")
            f.write("    list.add(" + name + ");\n")
    else:
        for nd in sample_input:
            shape = nd.shape
            dtype = str(nd.dtype).upper()
            f.write("    list.add(manager.ones(new Shape" + str(shape) + ", DataType." + dtype + "));\n")
    f.write("    System.out.println(predictor.predict(list));\n")
    f.write("  }\n}\n")


def zip_files(path: str, saved_path: str):
    files = os.listdir(path)
    dir_len = len(path)
    with zipfile.ZipFile(saved_path + ".zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            file_path = os.path.join(path, file)
            zipf.write(file_path, file_path[dir_len:])


def zip_dir(path: str, saved_path: str):
    shutil.make_archive(saved_path, 'zip', path)
