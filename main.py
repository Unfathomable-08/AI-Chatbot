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

# Data Loader Class
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# Model Class
class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, output_size):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(vocab_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Training and Testing Function
def training_testing_model():
    # Data Split
    X_train, y_train = train_test_split("train")
    X_test, y_test = train_test_split("test")
    X_val, v_test = train_test_split("val")
    num_outputs = len(set(y_train))

    # Datasets and DataLoaders
    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, v_test)
    test_dataset = SimpleDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize model and move to device
    model = FeedForwardNeuralNetwork(vocab_size=len(tokens), output_size=num_outputs).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    for epochs in range(100):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epochs % 5 == 0:
            print(f'Epoch {epochs}, Loss: {loss.item():.4f}')

training_testing_model()
