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

# Sparse Dataset
class SparseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.LongTensor(y).squeeze()
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        row = self.X[i].tocoo()
        indices = torch.LongTensor(np.vstack((row.row, row.col)))
        values = torch.FloatTensor(row.data)
        shape = torch.Size([1, self.X.shape[1]])  # Shape [1, 20278]
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
        return sparse_tensor, self.y[i]

# Custom collate function
def sparse_collate_fn(batch):
    X_batch = torch.stack([item[0].coalesce() for item in batch])  # Shape [16, 1, 20278]
    y_batch = torch.LongTensor([item[1] for item in batch])
    return X_batch, y_batch

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

# Data Preparation
print("Splitting the data...")
X_train, y_train = train_test_split("train")
X_test, y_test = train_test_split("test")
X_val, y_val = train_test_split("val")

# Reindex labels to 0-based
all_labels = sorted(set(y_train) | set(y_test) | set(y_val))
label_map = {old: new for new, old in enumerate(all_labels)}  # e.g., {1: 0, 2: 1, 3: 2, 4: 3}
y_train = [label_map[y] for y in y_train]
y_test = [label_map[y] for y in y_test]
y_val = [label_map[y] for y in y_val]

num_outputs = len(set(y_train))

print("Creating datasets...")
train_dataset = SparseDataset(X_train, y_train)
val_dataset = SparseDataset(X_val, y_val)
test_dataset = SparseDataset(X_test, y_test)

print("Creating dataloaders...")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True, collate_fn=sparse_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=2, pin_memory=True, collate_fn=sparse_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=2, pin_memory=True, collate_fn=sparse_collate_fn)

# Initialize model
print("Initializing model...")
model = FeedForwardNeuralNetwork(vocab_size=len(tokens), output_size=num_outputs).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def training_testing_model(epochs=10):
    print("Starting training loop...")
    for epoch in range(epochs):
        check_memory()
        model.train()
        print(f"\nEpoch {epoch + 1} / {epochs}")
        total_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 600 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
    print("Training completed.")

# Run training
training_testing_model()