import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string
from scipy.sparse import csr_matrix

# nltk.download("punkt")
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')

unwanted_tags = ['PRP', 'PRP$', 'DT', 'IN', 'CC', 'TO', 'MD', 'RP']

def dialog_separator(dataset):
    pairs = []
    for data in dataset:
        dialogs = data["dialog"]
        topics = data["topic"]
        for i in range(len(dialogs)):
            pairs.append((dialogs[i], topics))
    return pairs

def create_topic2idx(dataset):
    topics = set()
    for _, topic in dataset:
        topics.add(topic)  # Collect unique topics

    # Create a dictionary mapping each topic to a unique index
    topic2idx = {topic: idx for idx, topic in enumerate(sorted(topics))}
    return topic2idx

def tokenization(data):
    tokens = set()
    for dialog, _ in data:
        words = [w for w in word_tokenize(dialog.lower()) if w not in string.punctuation]
        pos_tags = pos_tag(words)
        tokens.update(word for word, tag in pos_tags if tag not in unwanted_tags)
    return list(tokens)

def bow_vector_sparse(sentence, word2idx):
    indices = []
    values = []
    for word in word_tokenize(sentence.lower()):
        if word in word2idx:
            indices.append(word2idx[word])
            values.append(1.0)
    return indices, values

def split_vector_sparse(data_pairs, word2idx, topic2idx):
    data = []
    row_indices = []
    col_indices = []
    y = []
    for i, (dialog, topic) in enumerate(data_pairs):
        indices, values = bow_vector_sparse(dialog, word2idx)
        data.extend(values)
        row_indices.extend([i] * len(values))
        col_indices.extend(indices)
        y.append(topic2idx.get(topic, -1))

    X = csr_matrix((data, (row_indices, col_indices)), shape=(len(data_pairs), len(word2idx)))
    return X, y