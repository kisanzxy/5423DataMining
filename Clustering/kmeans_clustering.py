
from sklearn.cluster import KMeans


class KmeansCluster:
    def __init__(self, K_num, label):

        self.name = "kmeans" + str(label)
        self.label = label
        self.K_num = K_num
        self.clusters = dict([])
        self.clustering = None
        self.feature_select_list = []

        
    def produce_clusters(self, feature_Docs):
        # Here are the corresponding classlabel and feature selected for the clusters
        for index in range(self.K_num):
            self.clusters[index] = []
        
        classlabel_input = []
        feature_docs_input = []
        
        # produce clusters
        for feature_doc in feature_Docs.values():
            classlabel_input.append(feature_doc.get_class_label(self.label))
            feature_docs_input.append(feature_doc.features_list)   # produce the local feature list
            self.feature_select_list.append(feature_doc)
            
            
        self.clustering = KMeans(init='k-means++', n_clusters = self.K_num)
        train_clusters = self.clustering.fit_predict(feature_docs_input)
        '''
            fit_predict(X) with X as the feature matrix and returns the cluster labels
        '''
        # separate all the feature vectors according to the clustering:
        for featureID, clusterID in enumerate(train_clusters):
        
            self.clusters[clusterID].append(self.feature_select_list[featureID])

        self.classlabel_set = set().union(*classlabel_input)