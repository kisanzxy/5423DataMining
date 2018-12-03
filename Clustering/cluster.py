
from feature import *
import time
import random
import os
from math import log
from dbscan_clustering import *
from kmeans_clustering import *


def clustering_exec(feature_list, label, EPSILON_list, minpts_list, K_num_list):
    outputKMEANSClustering(feature_list, label, K_num_list)
    outputDBSCANClustering(feature_list, label, EPSILON_list, minpts_list)
    
    
    
def outputKMEANSClustering(feature_list, label, K_num_list):
    filename = str(label) + '_Summary_KMEANS_Clustering'+ ".txt"
    write_data = open(os.path.join(os.getcwd(), filename), 'w')
    for K_num in K_num_list:
        clustering = KmeansCluster(K_num, label)
        
        write_data.write('=' * 30 + '\n')
        write_data.write("Clustering Parameters: K_NUM is " + str(K_num) + "\n")
        outputClusteringContents(write_data, clustering, feature_list, label)
    
    write_data.close()
 
def outputDBSCANClustering(feature_list, label, EPSILON_list, minpts_list):
    filename = str(label) + '_Summary_DBSCAN_Clustering'+ ".txt"
    write_data = open(os.path.join(os.getcwd(), filename), 'w')
    for EPSILON in EPSILON_list:
        for minpts in minpts_list:
            clustering = DBScanCluster(EPSILON, minpts, label)
            
            write_data.write('=' * 30 + '\n')
            write_data.write("Clustering Parameters: minpts is " + str(minpts) + " and EPSILON is " + str(EPSILON) + "\n")
            outputClusteringContents(write_data, clustering, feature_list, label)
       
        write_data.write('%' * 30 + '\n')
        write_data.write('The same EPSILON (Radius above)' + '\n')
    
    write_data.close()
    
    
### After obtain the desired cluster and start to write the contents
def outputClusteringContents(write_data, clustering, feature_list, label):
    for featureID, feature in enumerate(feature_list):
        valid_feature_doc = feature_select_valid_label(feature, label)
        
        print "\nClustering result: ", clustering.name, " for the feature ", feature.feature_name
        
        clustering.produce_clusters(valid_feature_doc)
        clustering_evaluation = ClusterEvaluate(valid_feature_doc, clustering, label)         
        write_data.write('_' * 30 + '\n')
        write_data.write("Clustering result: " + str(clustering.name) + " for the feature " + str(feature.feature_name) + "\n")
        
        write_data.write("The total number of valid article with label " + str(label) + " is: " + str(len(valid_feature_doc)) + "\n")
        write_data.write("SSE of the clustering is : " + str(clustering_evaluation.SSE) + "\n")
        write_data.write("Entropy of non-clustering is : " + str(clustering_evaluation.raw_Entropy) + "\n")
        write_data.write("Entropy after clustering is : " + str(clustering_evaluation.clustering_Entropy) + "\n")
        write_data.write("Information gain from entropy is : " + str(clustering_evaluation.info_gain_entropy) + "\n")
        write_data.write('\n')
    
class ClusterEvaluate:
    def __init__(self,feature_doc, clustering, label):
        self.docLength = float(len(feature_doc))
        self.clustering = clustering
        self.raw_Entropy = self.calculate_Entropy({0 : clustering.feature_select_list}, label)
        self.clustering_Entropy = self.calculate_Entropy(clustering.clusters, label)
        self.info_gain_entropy = self.raw_Entropy - self.clustering_Entropy
        self.SSE = self.calculate_SSE(clustering.clusters)
        
    def calculate_SSE(self, clusters):

        sumOfCount = 0.0
        sumOfVariance = 0.0

        for cluster in clusters.values():
            sumOfCount += len(cluster)
        mean = sumOfCount / len(clusters)
        for cluster in clusters.values():
            sumOfVariance += (len(cluster) - mean) ** 2
        variance = sumOfVariance / len(clusters)
        return variance
    
    
    def calculate_Entropy(self, clusters, label):
        entropy = 0.0
        
        # extract each vector of cluster labels:
        for cluster in clusters.values():
            # print cluster
            clusterLength = float(len(cluster))
            weight_entropy = clusterLength / self.docLength
            # calculate entropy of each cluster
            cluster_entropy = 0.0
            for labelContent in self.clustering.classlabel_set:
                factor = 0.0
                for select_feature in cluster:
                    if labelContent in select_feature.get_class_label(label):
                        factor += 1.0
                if factor > 0:
                    factor /= clusterLength
                    cluster_entropy += -factor * log(factor, 2)
            entropy += weight_entropy * cluster_entropy
        return entropy

    
            