
from Preprocessing import *
# from tfidf_weight import *

from sklearn_weight import *
from feature import *
from output_functions import *
from cluster import *
import time

###############################################################################
########################### main function ####################################

def main(argv):
    print('Generating document objects. This may take some time...')
    
    preprocess_start = time.time()
    trie = Trie()
    Total_Doc = read_documents(trie)    # a list of all the articles
    weightObj = Weights_sklearn(Total_Doc)
    feature_list = generate_multi_features(Total_Doc, weightObj, [5], [1])
    preprocess_period = time.time() - preprocess_start
    print 'The time spent on preprocessing is: ', preprocess_period/60.0, ' minutes'
    
    labels = ['places', 'topics']
    EPSILON_list = [0.1, 0.3, 1]
    minpts_list = [6, 10]
    K_num_list = [6, 10]
    
    clustering_start = time.time()
    for label in labels:  
        clustering_exec(feature_list, label, EPSILON_list, minpts_list, K_num_list)
        
    clustering_period = time.time() - clustering_start  
    print 'The time spent on clustering is: ', clustering_period/60.0, ' minutes' 
    
    print 'The count of objects with no entries for topics is: ', count_term_val(Total_Doc, 'topics', [])
    print 'The count of objects with no entries for places is: ', count_term_val(Total_Doc, 'places', [])
    
if __name__ == "__main__":
    main(sys.argv[1:])
