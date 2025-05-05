from data_proceeding import dialog_separator, tokenization, split_vector_sparse, create_topic2idx
from data_load import data, wiki_data, sciq_data

# Genarate pairs of dialog, topic in all datsets
qa_pairs = dialog_separator(data)
wikiqa_pairs = dialog_separator(wiki_data)
sciq_pairs = dialog_separator(sciq_data)

# Join all datasets
all_pairs = qa_pairs + wikiqa_pairs + sciq_pairs

# Tokenize Datasets
tokens = tokenization(all_pairs)

# Encoding words
word2idx = {word: idx for idx, word in enumerate(tokens)}

# Encoding topics
topic2idx = create_topic2idx(all_pairs)

# Split into X, y
def xy_split():
    X, y = split_vector_sparse(all_pairs, word2idx, topic2idx)
    return X, y
