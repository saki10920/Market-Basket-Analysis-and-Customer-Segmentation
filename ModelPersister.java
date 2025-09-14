
package project;

import weka.core.SerializationHelper;
import weka.classifiers.Classifier;

public class ModelPersister {
    
    public void saveModel(Classifier model, String filename) throws Exception {
        SerializationHelper.write(filename, model);
        System.out.println("Model saved to: " + filename);
    }
    
    public Classifier loadModel(String filename) throws Exception {
        Classifier model = (Classifier) SerializationHelper.read(filename);
        System.out.println("Model loaded from: " + filename);
        return model;
    }
}