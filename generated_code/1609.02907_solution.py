
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

num_nodes    = 200
num_edges    = 800
num_features = 16
num_classes  = 2

X          = torch.randn(num_nodes, num_features)
y          = torch.randint(0, num_classes, (num_nodes,))
edge_index = torch.randint(0, num_nodes, (2, num_edges))
adj        = torch.zeros(num_nodes, num_nodes)
adj[edge_index[0], edge_index[1]] = 1.0
adj[edge_index[1], edge_index[0]] = 1.0
adj       += torch.eye(num_nodes)
deg        = adj.sum(dim=1, keepdim=True).clamp(min=1)
adj        = adj / deg

perm       = torch.randperm(num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[perm[:160]] = True
test_mask[perm[160:]]  = True

class GraphConvNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GraphConvNet, self).__init__()
        self.layers = 2
        self.hidden_size = 64
        self.use_attention = False

        self.conv_layers = nn.ModuleList([nn.Linear(input_dim, self.hidden_size)])
        for _ in range(self.layers - 1):
            self.conv_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.output_layer = nn.Linear(self.hidden_size, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)

    def forward(self, x, adj):
        for i, conv_layer in enumerate(self.conv_layers):
            x = torch.matmul(adj, x)
            x = conv_layer(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.batch_norm(x)
        output = self.output_layer(x)
        return F.softmax(output, dim=1)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = GraphConvNet(num_features, num_classes).to(device)
X, y, adj = X.to(device), y.to(device), adj.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out  = model(X, adj)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1}/200 | Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    out      = model(X, adj)
    preds    = out[test_mask].argmax(dim=1)
    correct  = (preds == y[test_mask]).sum().item()
    total    = test_mask.sum().item()
    accuracy = 100.0 * correct / total

print(f"Final Accuracy: {accuracy:.2f}%")
