package ai.djl.examples;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.*;

import java.nio.file.Paths;

public class Example {
  public static void main(String[] args) throws Exception {

    Criteria<NDList, NDList> criteria = Criteria.builder()
      .setTypes(NDList.class, NDList.class)
      .optModelPath(Paths.get(args[0]))
      .build();
    ZooModel<NDList, NDList> model = ModelZoo.loadModel(criteria);
    Predictor<NDList, NDList> predictor = model.newPredictor();
    NDManager manager = NDManager.newBaseManager();
    NDList list = new NDList();
