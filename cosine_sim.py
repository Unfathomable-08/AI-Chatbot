from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix
from data_proceeding import dialog_separator, tokenization, split_vector_sparse, bow_vector_sparse
from data_handler import train_test_split, word2idx, train_pairs

def compute_cosine_similarity(input_sentence):
    # Convert input sentence to sparse vector
    idx, val = bow_vector_sparse(input_sentence, word2idx)
    input_vec = csr_matrix((val, ([0] * len(idx), idx)), shape=(1, len(word2idx)))

    # Prepare dataset matrix
    X, _ = train_test_split()

    # Compute cosine similarities
    similarities = cosine_similarity(input_vec, X).flatten()

    # Get index of most similar sentence
    best_idx = similarities.argmax()
    return train_pairs[best_idx + 1][0]

while True:
    dialog = input("Enter dialog (or 'quit' to exit): ")
    if dialog.lower() == 'quit':
        break
    predicted = compute_cosine_similarity(dialog)
    if predicted is not None:
        print(predicted)