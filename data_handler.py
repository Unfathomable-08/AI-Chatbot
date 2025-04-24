# Load and process dataset
from data_load import dataset
from data_proceeding import dialog_separator, tokenization, split_vector_sparse
train = dataset["train"]
test = dataset["test"]
validation = dataset["validation"]

train_pairs = dialog_separator(train)
test_pairs = dialog_separator(test)
validation_pairs = dialog_separator(validation)

tokens = tokenization(train_pairs)
word2idx = {word: idx for idx, word in enumerate(tokens)}

def train_test_split(data):
    if data == "train":
        X, y = split_vector_sparse(train_pairs, word2idx)
    elif data == "test":
        X, y = split_vector_sparse(test_pairs, word2idx)
    elif data == "val":
        X, y = split_vector_sparse(validation_pairs, word2idx)
    return X, y