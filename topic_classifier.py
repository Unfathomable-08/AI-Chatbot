from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
import numpy as np

from data_handler import xy_split
from data_proceeding import bow_vector_sparse
from data_handler import word2idx, topic2idx

# Load data and split
X, y = xy_split()
y = np.array(y).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB(alpha=0.01)  # Updated to best alpha
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Prediction function
def predict_topic(sentence):
    idx, val = bow_vector_sparse(sentence, word2idx)
    input_vec = csr_matrix((val, ([0] * len(idx), idx)), shape=(1, len(word2idx)))
    pred = model.predict(input_vec)
    return next(key for key, value in topic2idx.items() if value == pred[0])