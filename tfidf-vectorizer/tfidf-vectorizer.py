import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    N = len(documents)

    # Tokenize
    tokenized_docs = []
    for doc in documents:
        doc = doc.lower()
        words = doc.split()
        tokenized_docs.append(words)

    # Build vocab
    vocabulary = []
    all_word = []
    for doc in tokenized_docs:
        for word in doc:
            all_word.append(word)
    vocabulary = sorted(set(all_word))
    
    vocab_index = {}
    for idx, word in enumerate(vocabulary):
        vocab_index[word] = idx

    # Compute document frequency
    df = Counter()
    for doc in tokenized_docs:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] += 1

    # Compute IDF
    idf = {}
    for term in vocabulary:
        idf[term] = math.log(N/df[term])

    # Compute TF-IDF
    tfidf_matrix = np.zeros((N, len(vocabulary)))
    for i, doc in enumerate(tokenized_docs):
        term_counts = Counter(doc)
        total_terms = len(doc)
        for term, count in term_counts.items():
            tf = count / total_terms
            tfidf_matrix[i][vocab_index[term]] = tf * idf[term]
    
    return tfidf_matrix, vocabulary
    pass