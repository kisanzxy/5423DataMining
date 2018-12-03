
from Preprocessing import *
# from tfidf_weight import *

from sklearn_weight import *
from feature import *
from output_functions import *
from classifier import *

###############################################################################
########################### main function ####################################

def main(argv):
   
    # generate list of document objects for feature selection
    print('Generating document objects. This may take some time...')
    # establish a trie with all meaningful words in lowercase for all the articles
    trie = Trie()
    Total_Doc = read_documents(trie)    # a list of all the articles
    # wordinfo = wordInfo()
    # wordinfo.generate_freq_doc(documents, trie)
    
    # debug_output(wordinfo.articleLength, 'articleLength')
    # debug_output(wordinfo.articleFreq, 'articleWordFreq_normalized')
    
#    weightObj = Weights_sklearn(Total_Doc)
#    feature_list = generate_multi_features(Total_Doc, weightObj, [5], [10])
#    
#    classify_exec(feature_list, 'topics')
    # print feature.select_features
    
    
    # debug_output(weightFactors, 'weightFactors')
    
    print 'The count of objects with no entries for topics is: ', count_term_val(Total_Doc, 'topics', [])
    print 'The count of objects with no entries for places is: ', count_term_val(Total_Doc, 'places', [])
    
    write_output(Total_Doc, "topics")
    write_output(Total_Doc, "places")
    
        
    with open("count.txt", "a") as myfile:
        myfile.write("The count of objects with no entries for topics is: ")
        myfile.write(str(count_term_val(Total_Doc, 'topics', [])))
        myfile.write("\nThe count of objects with no entries for places is: ")
        myfile.write(str(count_term_val(Total_Doc, 'places', [])))

if __name__ == "__main__":
    main(sys.argv[1:])