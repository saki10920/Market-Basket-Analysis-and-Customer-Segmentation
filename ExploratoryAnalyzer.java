
package project;

import weka.core.Instances;
import weka.core.AttributeStats;

public class ExploratoryAnalyzer {
    
    public void performAnalysis(Instances dataset) {
        System.out.println("\n=== EXPLORATORY ANALYSIS ===");
        
        // Basic stat
        printBasicStatistics(dataset);
        
        // eta class distribution
        printClassDistribution(dataset);
        
        // attribute correlations eta
        printAttributeInfo(dataset);
    }
    
    private void printBasicStatistics(Instances dataset) {
        System.out.println("Dataset statistics:");
        for (int i = 0; i < dataset.numAttributes(); i++) {
            AttributeStats stats = dataset.attributeStats(i);
            if (dataset.attribute(i).isNumeric()) {
                System.out.println(dataset.attribute(i).name() + ": " +
                                 "mean=" + String.format("%.4f", stats.numericStats.mean) + ", " +
                                 "std=" + String.format("%.4f", stats.numericStats.stdDev) + ", " +
                                 "min=" + String.format("%.4f", stats.numericStats.min) + ", " +
                                 "max=" + String.format("%.4f", stats.numericStats.max));
            }
        }
    }
    
    private void printClassDistribution(Instances dataset) {
        if (dataset.classAttribute().isNominal()) {
            System.out.println("\nClass distribution:");
            int[] classCounts = dataset.attributeStats(dataset.classIndex()).nominalCounts;
            for (int i = 0; i < classCounts.length; i++) {
                double percentage = (double) classCounts[i] / dataset.numInstances() * 100;
                System.out.println(dataset.classAttribute().value(i) + ": " + 
                                  classCounts[i] + " (" + String.format("%.2f", percentage) + "%)");
            }
        }
    }
    
    private void printAttributeInfo(Instances dataset) {
        System.out.println("\nAttribute information:");
        System.out.println("Number of instances: " + dataset.numInstances());
        System.out.println("Number of attributes: " + dataset.numAttributes());
        System.out.println("Class attribute: " + dataset.classAttribute().name());
    }
    
    public boolean hasClassImbalance(Instances dataset) {
        if (!dataset.classAttribute().isNominal()) return false;
        
        int[] classCounts = dataset.attributeStats(dataset.classIndex()).nominalCounts;
        int maxCount = 0;
        int minCount = Integer.MAX_VALUE;
        
        for (int count : classCounts) {
            if (count > maxCount) maxCount = count;
            if (count < minCount) minCount = count;
        }
        
        // imbalance consider korbo if ratio > 5:1
        return (double)maxCount / minCount > 5;
    }
}