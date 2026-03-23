
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

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

num_classes = len(np.unique(y))
input_dim   = X_train.shape[1]

class LSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = 128
        self.layers = 1
        self.use_attention = False
        self.fc = nn.Linear(input_dim, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.layers, batch_first=True)
        if self.use_attention:
            self.attn = nn.MultiheadAttention(self.hidden_size, num_heads=1, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc(x))
        if self.use_attention:
            x = x.unsqueeze(1)
            out, _ = self.attn(x, x, x)
            x = out.squeeze(1)
        else:
            x = x.unsqueeze(1)
            x, _ = self.lstm(x)
            x = x.squeeze(1)
        x = self.fc_out(x)
        return x

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = LSTM(input_dim, num_classes).to(device)
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
