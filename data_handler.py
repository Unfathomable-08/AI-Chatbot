from data_proceeding import dialog_separator, tokenization, split_vector_sparse
from data_load import data

train_pairs = dialog_separator(data)

tokens = tokenization(train_pairs)
word2idx = {word: idx for idx, word in enumerate(tokens)}
act2idx = {
    "greeting": 0,
    "inquiry": 1,
    "farewell": 2,
    "compliment": 3,
    "gratitude": 4,
    "apology": 5,
    "request": 6,
    "confirmation": 7,
    "suggestion": 8,
    "expression": 9
}

def train_test_split():
    X, y = split_vector_sparse(train_pairs, word2idx, act2idx)
    return X, y