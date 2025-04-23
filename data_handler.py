from data_load import dataset
from data_proceeding import dialog_separator
from data_proceeding import tokenization
import numpy as np
from nltk.tokenize import word_tokenize

# Tokenize already existing dataset
train = dataset["train"]
test = dataset["test"]
validation = dataset["validation"]

train_pairs = dialog_separator(train)
test_pairs = dialog_separator(test)
validation_pairs = dialog_separator(validation)

tokens = tokenization(train_pairs)

# Assign a unique index to each token
word2idx = {word: idx for idx, word in enumerate(tokens)}

# Bag of Words Encoding Function
def bow_vector(sentence):
    vector = np.zeros(len(word2idx), dtype=np.float32)

    for word in word_tokenize(sentence.lower()): #tokenize the sentence provided
        if word in word2idx: #ignore words that are not in tokens
            vector[word2idx[word]] += 1.0  

    return vector

# X, y spliting
def split_vector(data_pairs):
    X = [bow_vector(dialog) for dialog, act in data_pairs]
    y = [act for dialog, act in data_pairs]

    return X, y

# Train Test Val Spliting
def train_test_split(data):
    if data == "train":
        X, y = split_vector(train_pairs) 
    elif data == "test":
        X, y = split_vector(test_pairs)  
    elif data == "val":
        X, y = split_vector(validation_pairs)  

    return X, y