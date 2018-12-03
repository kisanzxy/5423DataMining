
import os
import string

def count_term_val(documents, term, val):
    termCount = 0
    for document in documents:
        if document.get_term(term) == val:
            termCount += 1
    return termCount

# term is just the string: topic or places
def write_output(documents, term):
    filename = term + ".txt"
    write_data = open(os.path.join(os.getcwd(), filename), 'w')
    termList = []
    for document in documents:
        termList.append(document.get_term(term))
    
    write_data.write(str(termList))
    write_data.close()

def debug_output(debug_contents, debug_name):
    filename = debug_name + ".txt"
    write_data = open(os.path.join(os.getcwd(), filename), 'w')
    write_data.write(str(debug_contents))
    write_data.close()