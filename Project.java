
package project;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Project {
    
    public static void main(String[] args) {
        try {
            DataSource source = new DataSource("C:\\Users\\masud\\OneDrive\\Desktop\\Project\\data\\groceries_binary.arff");
            Instances dataset = source.getDataSet();
            dataset.setClassIndex(dataset.numAttributes() - 1);
            
            DataPreprocessor preprocessor = new DataPreprocessor();
            Instances processedData = preprocessor.preprocessData(dataset);
            
            ExploratoryAnalyzer analyzer = new ExploratoryAnalyzer();
            analyzer.performAnalysis(processedData);
            
            SupervisedLearner supervisedLearner = new SupervisedLearner();
            supervisedLearner.trainAndEvaluate(processedData);
            

            UnsupervisedLearner unsupervisedLearner = new UnsupervisedLearner();
            unsupervisedLearner.performClustering(processedData);
            
            AssociationMiner associationMiner = new AssociationMiner();
            associationMiner.mineAssociationRules(processedData);
            

            ModelPersister modelPersister = new ModelPersister();
            weka.classifiers.trees.RandomForest bestModel = new weka.classifiers.trees.RandomForest();
            bestModel.buildClassifier(processedData);
            modelPersister.saveModel(bestModel, "best_model.model");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}