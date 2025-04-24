import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_handler import train_test_split, tokens

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