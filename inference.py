import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_handler import train_test_split, tokens, word2idx
from data_proceeding import bow_vector_sparse
from scipy.sparse import csr_matrix

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model
class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, output_size):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(vocab_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = x.to_dense().squeeze(1)  # [16, 1, 20278] → [16, 20278]
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Load Model
model = FeedForwardNeuralNetwork(vocab_size=len(tokens), output_size=4)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.to(device)
model.eval()




def predict_dialog_act(dialog, word2idx, model, device, vocab_size):
    try:
        # Create sparse BoW representation
        indices, values = bow_vector_sparse(dialog, word2idx)
        if not indices:
            print("No valid words found in dialog.")
            return None
        
        # Create csr_matrix
        row_indices = [0] * len(indices)
        X = csr_matrix((values, (row_indices, indices)), shape=(1, vocab_size))
        
        # Convert to sparse tensor
        row = X.tocoo()
        indices = torch.LongTensor(np.vstack((row.row, row.col)))
        values = torch.FloatTensor(row.data)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size([1, vocab_size]))
        sparse_tensor = sparse_tensor.to(device)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            y_pred = model(sparse_tensor.unsqueeze(0))  # [1, 1, vocab_size] -> [1, 4]
            _, predicted = torch.max(y_pred, 1)
        
        # Return original label (1, 2, 3, 4)
        return predicted.item() + 1  # Add 1 to map {0, 1, 2, 3} to {1, 2, 3, 4}
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

dialog = input("Enter dialog (or 'quit' to exit): ")

predicted_act = predict_dialog_act(dialog, word2idx, model, device, vocab_size=len(tokens))
if predicted_act is not None:
    print(f"Predicted dialog act (original): {predicted_act}")