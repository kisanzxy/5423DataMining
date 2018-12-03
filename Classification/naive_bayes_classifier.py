
from sklearn.naive_bayes import MultinomialNB

class Naive_Bayes:
    def __init__(self, EPSILON, label):
        self.name = 'NBCLF_for_class_' + str(label)
        self.label = label
        self.EPSILON = EPSILON
        self.classify = None
        self.classlabel_input = []
    
    
    def train_classifier(self, training_feature_docs):
        mnb_classifier = MultinomialNB()
        self.classlabel_input = []
        feature_docs_input = []
        
        for feature_doc in training_feature_docs:
            self.classlabel_input.append(feature_doc.get_class_label(self.label))
            feature_docs_input.append(feature_doc.features_list)
            
        self.classify = mnb_classifier.fit(feature_docs_input, self.classlabel_input)
    
    def test_classifier(self, testing_features_doc):
        correct_count = 0.0
        
        for feature_doc in testing_features_doc:
            result_label = self.test_one_feature_doc(feature_doc)
            if set(result_label).intersection(set(feature_doc.get_class_label(self.label))):
                correct_count += 1.0
        return correct_count
        
        
    def test_one_feature_doc(self, feature_doc):
        result_label = []
        prob = self.classify.predict_proba([feature_doc.features_list])
        
        for index, each_prob in enumerate(prob[0]):
            if each_prob > self.EPSILON:
                result_label += self.classlabel_input[index]
                
        return result_label
        
        