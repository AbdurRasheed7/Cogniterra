
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(42)
np.random.seed(42)

print("Loading 20newsgroups dataset...")
data   = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), data_home='/app/data')
texts  = data.data
labels = data.target

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(texts).toarray().astype(np.float32)
y = np.array(labels, dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fast mode — subset data before creating tensors
if True:
    X_train, y_train = X_train[:5000], y_train[:5000]
    X_test,  y_test  = X_test[:1000],  y_test[:1000]

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

num_classes = len(np.unique(y))
input_dim   = X_train.shape[1]

class Transformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Transformer, self).__init__()
        self.hidden_size = 512
        self.num_heads = 8
        self.num_layers = 6
        self.input_projection = nn.Linear(input_dim, self.hidden_size)
        self.layers = nn.ModuleList([nn.ModuleList([nn.MultiheadAttention(self.hidden_size, self.num_heads, batch_first=True), 
                                                    nn.LayerNorm(self.hidden_size), 
                                                    nn.Linear(self.hidden_size, self.hidden_size), 
                                                    nn.LayerNorm(self.hidden_size)]) for _ in range(self.num_layers)])
        self.output_projection = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        for layer in self.layers:
            attention, layer_norm1, linear, layer_norm2 = layer
            x = x.unsqueeze(1)
            out, _ = attention(x, x, x)
            x = out.squeeze(1)
            x = layer_norm1(x + x)
            x = F.relu(x)
            x = linear(x)
            x = layer_norm2(x + x)
        x = self.output_projection(x)
        return x

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = Transformer(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs  = model(X_batch)
        _, preds = torch.max(outputs, 1)
        total   += y_batch.size(0)
        correct += (preds == y_batch).sum().item()

accuracy = 100.0 * correct / total
print(f"Final Accuracy: {accuracy:.2f}%")
