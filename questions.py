import nltk
import sys
import os
import string
import math

nltk.download('stopwords')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    files = dict()
    path = os.path.join(os.getcwd(), directory)
    print(path)
    for file in os.listdir(path):
        f = open(os.path.join(path, file), 'r')
        files[file] = f.read()

    return files
    

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    final_tokens = []
    tokens = nltk.tokenize.word_tokenize(document.lower())
    for token in tokens:
        if token not in nltk.corpus.stopwords.words("english"):
            if token not in string.punctuation:
                final_tokens.append(token)

    return final_tokens



def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    num_docs = len(documents)
    idf_dict = dict()
    
    for doc in documents:
        doc_words = set(documents[doc])
        for word in doc_words:
            if word not in idf_dict:
                idf_dict[word] = 1
            else:
                idf_dict[word] += 1

    for word in idf_dict:
        idf_dict[word] = math.log((num_docs / idf_dict[word]))
    
    return idf_dict

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_idfs = {file:0 for file in files}
    for word in query:
        if word in idfs:
            for file in files:
                tf = files[file].count(word)
                tf_idf = tf * idfs[word]
                file_idfs[file] += tf_idf

    sorted_files = sorted([file for file in files], key= lambda x:file_idfs[x], reverse=True)

    return sorted_files[:n]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_score = {sentence:{'idf': 0, 'len': 0, 'query_count': 0, 'qtd': 0} for sentence in sentences}
    for sentence in sentences:
        s = sentence_score[sentence]
        s['len'] = len(sentences[sentence])
        for word in query:
            if word in sentences[sentence]:
                s['query_count'] += sentences[sentence].count(word)
                s['idf'] += idfs[word]
        s['qtd'] = s['query_count'] / s['len']
    
    sorted_sentences = sorted([sentence for sentence in sentences], key=lambda x: (sentence_score[x]['idf'], sentence_score[x]['qtd']), reverse=True)

    return sorted_sentences[:n]

if __name__ == "__main__":
    main()

# document = "Would you like to see my cat? I met it a few days back and it is so cute."
# document = "You are good my love. i like. to see it"

