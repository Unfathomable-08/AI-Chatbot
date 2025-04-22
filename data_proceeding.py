import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import string

# nltk.download("punkt")
# nltk.download('punkt_tab')

# Separating into dialog and next act
def dialog_separator(dataset):
    pairs = []
    for data in dataset:
        dialogs = data["dialog"]
        acts = data["act"] 
        for i in range(len(dialogs) - 1):
            pairs.append((dialogs[i], acts[i + 1])) # Current dialog and next act
    
    return pairs

# Tokenizing the dialogs
def tokenization(data):
    tokenized_words_puncts = [word_tokenize(dialog.lower()) for dialog, act in data]

    tokenized_words = [[word for word in tokens if word not in string.punctuation] for tokens in tokenized_words_puncts]

    concatenated_tokens = [word for sublist in tokenized_words for word in sublist]

    tokens = list(set(concatenated_tokens))

    return tokens