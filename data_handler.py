from data_proceeding import dialog_separator, tokenization, split_vector_sparse
from data_load import data, wiki_data, sciq_data

qa_pairs = dialog_separator(data)
wikiqa_pairs = dialog_separator(wiki_data)
sciq_pairs = dialog_separator(sciq_data)

all_pairs = qa_pairs + wikiqa_pairs + sciq_pairs

tokens = tokenization(all_pairs)

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
    "expression": 9,
    "question": 10,
    "answer": 11
}

def train_test_split():
    X, y = split_vector_sparse(all_pairs, word2idx, act2idx)
    return X, y