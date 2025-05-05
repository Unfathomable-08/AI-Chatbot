from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix
from data_proceeding import dialog_separator, tokenization, split_vector_sparse, bow_vector_sparse
from data_handler import xy_split, word2idx, all_pairs
# from topic_classifier import predict_topic

def compute_cosine_similarity(input_sentence):
    # Convert input sentence to sparse vector
    idx, val = bow_vector_sparse(input_sentence, word2idx)
    input_vec = csr_matrix((val, ([0] * len(idx), idx)), shape=(1, len(word2idx)))

    # Prepare dataset matrix
    X, _ = xy_split()

    # Compute cosine similarities
    similarities = cosine_similarity(input_vec, X).flatten()

    # Get index of most similar sentence
    best_idx = similarities.argmax()
    return all_pairs[best_idx + 1][0]