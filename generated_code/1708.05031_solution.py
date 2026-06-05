
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import math
torch.manual_seed(42)
np.random.seed(42)

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
train_loader  = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=5, shuffle=False)


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=5):
        super(MatrixFactorization, self).__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        self.user_bias  = nn.Embedding(num_users, 1)
        self.item_bias  = nn.Embedding(num_items, 1)
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        u  = self.user_embed(user)
        i  = self.item_embed(item)
        ub = self.user_bias(user).squeeze()
        ib = self.item_bias(item).squeeze()
        return (u * i).sum(dim=1) + ub + ib


device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = MatrixFactorization(num_users, num_items).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for users, items, rts in train_loader:
        users, items, rts = users.to(device), items.to(device), rts.to(device)
        optimizer.zero_grad()
        preds = model(users, items)
        loss  = criterion(preds, rts)
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
        sq_err += ((preds - rts) ** 2).sum().item()
        count  += rts.size(0)

rmse     = math.sqrt(sq_err / count)
accuracy = max(0.0, 100.0 - (rmse * 20))
print(f"RMSE: {rmse:.4f}")
print(f"Final Accuracy: {accuracy:.2f}%")
