import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string

nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Define unwanted Part Of Speech (POS) tags
unwanted_tags = ['PRP', 'PRP$', 'DT', 'IN', 'CC', 'TO', 'MD', 'RP']

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

    # Remove punctuation
    tokenized_words = [[word for word in tokens if word not in string.punctuation] for tokens in tokenized_words_puncts]

    # Flatten the list
    concatenated_tokens = [word for sublist in tokenized_words for word in sublist]

    # POS tagging
    pos_tags = pos_tag(concatenated_tokens)

    # Filter tokens based on unwanted POS tags
    filtered_tokens = [word for word, tag in pos_tags if tag not in unwanted_tags]

    # Remove duplicates
    tokens = list(set(filtered_tokens))

    return tokens