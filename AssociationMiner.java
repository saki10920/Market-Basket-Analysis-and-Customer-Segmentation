package project;

import weka.core.Instances;
import weka.associations.Apriori;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import java.util.Random;

public class AssociationMiner {
    
    public void mineAssociationRules(Instances data) throws Exception {
        System.out.println("\n=== ASSOCIATION RULE MINING (USING ALL INSTANCES) ===");
        
        // Check if data is valid
        if (data == null || data.numInstances() == 0 || data.numAttributes() == 0) {
            System.out.println("Invalid data for association mining");
            return;
        }
        
        System.out.println("Using all " + data.numInstances() + " instances for mining");
        
        //Optimize data for mining 
        Instances optimizedData = optimizeDataForMining(data);
        
        // Check optimized data is still valid
        if (optimizedData == null || optimizedData.numInstances() == 0 || optimizedData.numAttributes() == 0) {
            System.out.println("No valid data after optimization for association mining");
            return;
        }
        
        System.out.println("Final dataset: " + optimizedData.numInstances() + " instances, " + 
                          optimizedData.numAttributes() + " attributes");
        
        // mine with memory-friendly parameters
        mineWithMemoryConstraints(optimizedData);
    }
    
    private Instances optimizeDataForMining(Instances originalData) throws Exception {
        System.out.println("Optimizing data for memory-efficient mining...");
        System.out.println("Original data: " + originalData.numInstances() + " instances, " + 
                          originalData.numAttributes() + " attributes");
        
        // use ALL instances 
        Instances sampleData = originalData;
        
        // Reduce dimensionality by selecting key attributes
        sampleData = selectKeyAttributes(sampleData);
        System.out.println("After attribute selection: " + sampleData.numAttributes() + " attributes");
        
        // Check we still have attributes
        if (sampleData.numAttributes() == 0) {
            System.out.println("Warning: All attributes were removed. Using original attributes.");
            sampleData = originalData; // Fallback to original
        }
        
        // Simple discretization with few bins (only if we have numeric attributes)
        if (hasNumericAttributes(sampleData)) {
            sampleData = simpleDiscretize(sampleData);
        } else {
            System.out.println("No numeric attributes to discretize");
        }
        
        return sampleData;
    }
    
    private Instances selectKeyAttributes(Instances data) throws Exception {
        System.out.println("Selecting key attributes for mining...");
        
        // subset of attributes gula ke rakhbe to reduce dimensionality but prevent memory issues
        StringBuilder attributesToKeep = new StringBuilder();
        int maxAttributes = Math.min(8, data.numAttributes()); // Keep max 8 attributes 
        

        boolean classIncluded = false;
        int attributesAdded = 0;
        
  
        if (data.classIndex() >= 0) {
            attributesToKeep.append((data.classIndex() + 1));
            classIncluded = true;
            attributesAdded++;
        }
        
        // Add other attributes 
        for (int i = 0; i < data.numAttributes() && attributesAdded < maxAttributes; i++) {
            if (i != data.classIndex()) { // Skip class 
                if (data.attribute(i).isNumeric()) { // Prefer numeric attributes
                    if (attributesToKeep.length() > 0) {
                        attributesToKeep.append(",");
                    }
                    attributesToKeep.append(i + 1);
                    attributesAdded++;
                }
            }
        }
        
        // add nominal attributes
        for (int i = 0; i < data.numAttributes() && attributesAdded < maxAttributes; i++) {
            if (i != data.classIndex() && !data.attribute(i).isNumeric()) {
                boolean alreadyAdded = false;
                String[] currentAttributes = attributesToKeep.toString().split(",");
                for (String attr : currentAttributes) {
                    if (attr.equals(String.valueOf(i + 1))) {
                        alreadyAdded = true;
                        break;
                    }
                }
                
                if (!alreadyAdded) {
                    if (attributesToKeep.length() > 0) {
                        attributesToKeep.append(",");
                    }
                    attributesToKeep.append(i + 1);
                    attributesAdded++;
                }
            }
        }
        
        System.out.println("Keeping attributes: " + attributesToKeep.toString());
        
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices(attributesToKeep.toString());
        removeFilter.setInvertSelection(true);
        removeFilter.setInputFormat(data);
        
        return Filter.useFilter(data, removeFilter);
    }
    
    private boolean hasNumericAttributes(Instances data) {
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isNumeric() && i != data.classIndex()) {
                return true;
            }
        }
        return false;
    }
    
    private Instances simpleDiscretize(Instances data) throws Exception {
        try {
            Discretize discretize = new Discretize();
            
            // Only discretize numeric attributes
            StringBuilder numericAttributes = new StringBuilder();
            for (int i = 0; i < data.numAttributes(); i++) {
                if (data.attribute(i).isNumeric() && i != data.classIndex()) {
                    if (numericAttributes.length() > 0) {
                        numericAttributes.append(",");
                    }
                    numericAttributes.append(i + 1); 
                }
            }
            
            if (numericAttributes.length() == 0) {
                System.out.println("No numeric attributes to discretize");
                return data;
            }
            
            discretize.setAttributeIndices(numericAttributes.toString());
            discretize.setBins(3); 
            discretize.setInputFormat(data);
            
            System.out.println("Discretizing " + numericAttributes.toString().split(",").length + " numeric attributes");
            return Filter.useFilter(data, discretize);
            
        } catch (Exception e) {
            System.out.println("Discretization failed: " + e.getMessage());
            System.out.println("Using original data without discretization");
            return data;
        }
    }
    
    private void mineWithMemoryConstraints(Instances data) throws Exception {
        System.gc(); // Force garbage collection  mining er age
        
        System.out.println("\nFinal data for mining: " + data.numInstances() + 
                          " instances, " + data.numAttributes() + " attributes");
        
        if (data.numInstances() < 10 || data.numAttributes() < 2) {
            System.out.println("Insufficient data for association mining");
            return;
        }
        
        // parameters adjust korsi based on dataset size
        double minSupport = calculateAppropriateSupport(data.numInstances());
        double minConfidence = 0.9; 
        
        System.out.println("Using minimum support: " + minSupport + ", confidence: " + minConfidence);
        
        System.out.println("\n1. General associations:");
        mineGeneralAssociations(data, minSupport, minConfidence);
        
        System.gc(); 
        
        System.out.println("\n2. Class associations:");
        mineClassAssociations(data, minSupport, minConfidence);
    }
    
    private double calculateAppropriateSupport(int numInstances) {
        
        if (numInstances > 10000) {
            return 0.3;
        } else if (numInstances > 5000) {
            return 0.25;
        } else if (numInstances > 1000) {
            return 0.2;
        } else {
            return 0.15; 
        }
    }
    
    private void mineGeneralAssociations(Instances data, double minSupport, double minConfidence) throws Exception {
        try {
            Apriori apriori = new Apriori();
            
            apriori.setOptions(new String[]{
                "-N", "10",           // Number of rules
                "-T", "1",           // Use lift
                "-C", String.valueOf(minConfidence),
                "-M", String.valueOf(minSupport),
                "-U", "1.0",
                "-S", "-1",
                "-c", "-1"           // No class
            });
            
            System.out.println("Mining general associations...");
            apriori.buildAssociations(data);
            System.out.println(apriori.toString());
            
        } catch (OutOfMemoryError e) {
            System.out.println("Out of memory during general association mining");
            System.out.println("Try increasing Java heap space: java -Xmx4g YourProgram");
        } catch (Exception e) {
            System.out.println("General association mining failed: " + e.getMessage());
        }
    }
    
    private void mineClassAssociations(Instances data, double minSupport, double minConfidence) throws Exception {
        try {
            // Only mine class associations if we have a class attribute
            if (data.classIndex() < 0) {
                System.out.println("No class attribute for class association mining");
                return;
            }
            
            Apriori apriori = new Apriori();
            
            apriori.setOptions(new String[]{
                "-N", "10",           // Number of rules
                "-T", "1",           
                "-C", String.valueOf(minConfidence + 0.05), // Slightly higher confidence for class rules
                "-M", String.valueOf(minSupport + 0.05),    // Slightly higher support for class rules
                "-U", "1.0",
                "-c", data.classIndex() + ""
            });
            
            System.out.println("Mining class associations...");
            apriori.buildAssociations(data);
            System.out.println(apriori.toString());
            
        } catch (OutOfMemoryError e) {
            System.out.println("Out of memory during class association mining");
        } catch (Exception e) {
            System.out.println("Class association mining failed: " + e.getMessage());
        }
    }
    
}