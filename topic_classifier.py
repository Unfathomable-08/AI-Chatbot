from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix

from data_handler import xy_split
from data_proceeding import bow_vector_sparse
from data_handler import word2idx

# Convert input sentence to sparse vector
idx, val = bow_vector_sparse("What is mercury atmosphere", word2idx)
input_vec = csr_matrix((val, ([0] * len(idx), idx)), shape=(1, len(word2idx)))
print(input_vec)

X, y = xy_split()

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgs = LogisticRegression()

pred = lgs.fit(X_train, y_train)
lgs.predict(input_vec)

print(pred)