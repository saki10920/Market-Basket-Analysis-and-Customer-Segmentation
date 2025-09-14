
package project;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;

public class DataPreprocessor {
    
    public Instances preprocessData(Instances dataset) throws Exception {
        Instances processedData = new Instances(dataset);
        
        // missing value handle kora hoise
        processedData = handleMissingValues(processedData);
        
        // string theke nominal kora hoise
        processedData = convertStringsToNominal(processedData);
        
        // numeric feature normalize
        processedData = simpleNormalize(processedData);
        
        System.out.println("Data preprocessing completed");
        return processedData;
    }
    
    private Instances handleMissingValues(Instances data) throws Exception {
        ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
        replaceMissing.setInputFormat(data);
        return Filter.useFilter(data, replaceMissing);
    }
    
    private Instances convertStringsToNominal(Instances data) throws Exception {
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isString() && i != data.classIndex()) {
                StringToNominal stringToNominal = new StringToNominal();
                stringToNominal.setInputFormat(data);
                stringToNominal.setAttributeRange("first-last");
                data = Filter.useFilter(data, stringToNominal);
            }
        }
        return data;
    }
    
    private Instances simpleNormalize(Instances data) throws Exception {
        Normalize normalize = new Normalize();
        normalize.setInputFormat(data);
        return Filter.useFilter(data, normalize);
    }
}
