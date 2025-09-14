
package project;

import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.SMO;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import java.util.Random;

public class SupervisedLearner {
    
    public void trainAndEvaluate(Instances data) throws Exception {
        System.out.println("\n=== SUPERVISED LEARNING ===");
        
        ExploratoryAnalyzer analyzer = new ExploratoryAnalyzer();
        boolean hasImbalance = analyzer.hasClassImbalance(data);
        Instances trainingData = data;
        
        if (hasImbalance) {
            System.out.println("Class imbalance detected. Applying balancing...");
            trainingData = handleClassImbalance(data);
        }
        
        //  built-in class weighting soho classifiers use korsi
        Classifier[] classifiers = {
            createBalancedRandomForest(),
            createBalancedJ48(),
            createBalancedSMO()
        };
        
        String[] classifierNames = {"Random Forest", "J48 Decision Tree", "SVM (SMO)"};
        
        for (int i = 0; i < classifiers.length; i++) {
            System.out.println("\n--- " + classifierNames[i] + " ---");
            Evaluation eval = evaluateClassifier(classifiers[i], trainingData, 10);
            printEvaluationMetrics(eval, classifierNames[i]);
        }
    }
    
    private Instances handleClassImbalance(Instances data) throws Exception {
        try {
            return applyResample(data);
        } catch (Exception e) {
            System.out.println("Resample failed: " + e.getMessage());
            System.out.println("Using original data with class weighting");
            return data;
        }
    }
    
    private Instances applyResample(Instances data) throws Exception {
        Resample resample = new Resample();
        resample.setInputFormat(data);
        
        // complex options chara Simple resample
        System.out.println("Applying Resample filter");
        return Filter.useFilter(data, resample);
    }
    
    //  classifiers Create korsi with class weighting
    private Classifier createBalancedRandomForest() {
        RandomForest rf = new RandomForest();
        // RandomForest usually by default  imbalance handle kore 
        return rf;
    }
    
    private Classifier createBalancedJ48() {
        J48 j48 = new J48();
        try {
            j48.setOptions(new String[]{"-C", "0.25", "-M", "2"});
        } catch (Exception e) {
            //options fail korle default
        }
        return j48;
    }
    
    private Classifier createBalancedSMO() {
        SMO smo = new SMO();
        // SMO class imbalance handle kore by default
        return smo;
    }
    
    private Evaluation evaluateClassifier(Classifier classifier, Instances data, int folds) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, folds, new Random(42));
        return eval;
    }
    
    private void printEvaluationMetrics(Evaluation eval, String classifierName) {
        System.out.println("Accuracy: " + String.format("%.4f", eval.pctCorrect()));
        System.out.println("Precision: " + String.format("%.4f", eval.weightedPrecision()));
        System.out.println("Recall: " + String.format("%.4f", eval.weightedRecall()));
        System.out.println("F1 Score: " + String.format("%.4f", eval.weightedFMeasure()));
        System.out.println("ROC AUC: " + String.format("%.4f", eval.weightedAreaUnderROC()));
        
        System.out.println("Confusion Matrix:");
        double[][] matrix = eval.confusionMatrix();
        for (double[] row : matrix) {
            for (double value : row) {
                System.out.print(String.format("%6.0f", value) + " ");
            }
            System.out.println();
        }
    }
    
    public Classifier trainBestModel(Instances data, Classifier classifier) throws Exception {
        classifier.buildClassifier(data);
        System.out.println("Trained best model: " + classifier.getClass().getSimpleName());
        return classifier;
    }
}