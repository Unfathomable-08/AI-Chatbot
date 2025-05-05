from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix
from data_proceeding import dialog_separator, tokenization, split_vector_sparse, bow_vector_sparse
from data_handler import xy_split, word2idx, all_pairs, topic2idx
from topic_classifier import predict_topic

def compute_cosine_similarity(input_sentence):
    predicted_topic = predict_topic(input_sentence)
    print(predicted_topic)

    # Convert input sentence to sparse vector
    idx, val = bow_vector_sparse(input_sentence, word2idx)
    input_vec = csr_matrix((val, ([0] * len(idx), idx)), shape=(1, len(word2idx)))

    # Filter when topic matches
    X_filtered = []
    X_index = []
    for i in range(X.shape[0]):  
        if y[i] == 6:
            X_filtered.append(X[i])
            X_index.append(i)

    X_filtered = vstack(X_filtered)

    # Compute cosine similarities
    similarities = cosine_similarity(input_vec, X_filtered).flatten()

    # Get index of most similar sentence
    best_idx = similarities.argmax()
    
    return all_pairs[X_index[best_idx + 1]][0]