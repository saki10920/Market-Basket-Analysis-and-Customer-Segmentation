
package project;

import weka.core.Instances;
import weka.clusterers.SimpleKMeans;

public class UnsupervisedLearner {
    
    public void performClustering(Instances data) throws Exception {
        System.out.println("\n=== UNSUPERVISED LEARNING ===");
        
        // clustering er jonno data ctreate korsi
        Instances clusterData = createClusterData(data);
        
        //  clustering perform
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(3);
        kmeans.buildClusterer(clusterData);
        
        System.out.println("K-means clustering completed with " + 
                          kmeans.getNumClusters() + " clusters");
        
        // cluster analysis
        analyzeBasicClusters(kmeans, clusterData);
    }
    
    private Instances createClusterData(Instances data) {
        Instances clusterData = new Instances(data);
        
        // class index is set and valid eta check
        if (clusterData.classIndex() >= 0 && 
            clusterData.classIndex() < clusterData.numAttributes()) {
            clusterData.setClassIndex(-1); // Remove class designation
        }
        
        return clusterData;
    }
    
    private void analyzeBasicClusters(SimpleKMeans kmeans, Instances clusterData) throws Exception {
        int[] clusterSizes = new int[kmeans.getNumClusters()];
        
        // instances count korsi for each cluster
        for (int i = 0; i < clusterData.numInstances(); i++) {
            int cluster = kmeans.clusterInstance(clusterData.instance(i));
            clusterSizes[cluster]++;
        }
        
        System.out.println("\nCluster sizes:");
        for (int i = 0; i < clusterSizes.length; i++) {
            double percentage = (double) clusterSizes[i] / clusterData.numInstances() * 100;
            System.out.println("Cluster " + i + ": " + clusterSizes[i] + 
                              " instances (" + String.format("%.1f", percentage) + "%)");
        }
    }
}
