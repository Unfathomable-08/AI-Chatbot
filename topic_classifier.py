# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from scipy.sparse import csr_matrix
# import numpy as np

# from data_handler import xy_split
# from data_proceeding import bow_vector_sparse
# from data_handler import word2idx

# # Convert input sentence to sparse vector
# idx, val = bow_vector_sparse("What is mercury atmosphere", word2idx)
# input_vec = csr_matrix((val, ([0] * len(idx), idx)), shape=(1, len(word2idx)))
# print(input_vec)

# X, y = xy_split()

# # Convert y to dense 1D array
# y = np.array(y).ravel()
# print(y[:5])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# lgs = LogisticRegression(solver='saga', max_iter=1000, n_jobs=-1)

# lgs.fit(X_train, y_train)


# y_pred = lgs.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))

# # Predict on custom input
# pred = lgs.predict(input_vec)
# print("\nPrediction for input sentence:", pred)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
import numpy as np

from data_handler import xy_split, word2idx
from data_proceeding import bow_vector_sparse

# Load data and split
X, y = xy_split()
y = np.array(y).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
lgs = LogisticRegression(solver='saga', max_iter=1000, n_jobs=-1)
lgs.fit(X_train, y_train)

# Evaluate
y_pred = lgs.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Prediction function
def predict_topic(sentence):
    idx, val = bow_vector_sparse(sentence, word2idx)
    input_vec = csr_matrix((val, ([0] * len(idx), idx)), shape=(1, len(word2idx)))
    prediction = lgs.predict(input_vec)
    return prediction[0]