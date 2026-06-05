import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.manual_seed(42)
np.random.seed(42)



import pandas as pd
from sklearn.model_selection import train_test_split
import math

print("Loading MovieLens 100K dataset...")
ratings = pd.read_csv('./data/ml-100k/u.data', sep='\t',
                      names=['user_id', 'item_id', 'rating', 'timestamp'])
num_users = ratings['user_id'].max() + 1
num_items = ratings['item_id'].max() + 1
print(f"Users: {num_users}, Items: {num_items}")

user_ids    = torch.tensor(ratings['user_id'].values, dtype=torch.long)
item_ids    = torch.tensor(ratings['item_id'].values, dtype=torch.long)
rating_vals = torch.tensor(ratings['rating'].values,  dtype=torch.float32)

indices  = list(range(len(ratings)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = TensorDataset(user_ids[train_idx], item_ids[train_idx], rating_vals[train_idx])
test_dataset  = TensorDataset(user_ids[test_idx],  item_ids[test_idx],  rating_vals[test_idx])
train_loader  = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=10, shuffle=False)

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=5):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.layers = nn.ModuleList([nn.Linear(embedding_dim * 2, embedding_dim) for _ in range(2)])
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, 1)  # added output layer

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embedding(user_ids)
        item_embeddings = self.item_embedding(item_ids)
        x = torch.cat((user_embeddings, item_embeddings), dim=1)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
            x = self.dropout(x)
            x = self.layer_norm(x)
        x = self.output_layer(x)  # added output layer
        return x

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = MatrixFactorization(num_users, num_items).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for users, items, rts in train_loader:
        users, items, rts = users.to(device), items.to(device), rts.to(device)
        optimizer.zero_grad()
        preds = model(users, items)
        loss  = criterion(preds, rts.view(-1, 1))  # changed rts to rts.view(-1, 1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")

model.eval()
sq_err = 0.0
count  = 0
with torch.no_grad():
    for users, items, rts in test_loader:
        users, items, rts = users.to(device), items.to(device), rts.to(device)
        preds   = model(users, items)
        sq_err += ((preds - rts.view(-1, 1)) ** 2).sum().item()  # changed rts to rts.view(-1, 1)
        count  += rts.size(0)

rmse     = math.sqrt(sq_err / count)
accuracy = max(0.0, 100.0 - (rmse * 20))
print(f"RMSE: {rmse:.4f}")
print(f"Final Accuracy: {accuracy:.2f}%")
