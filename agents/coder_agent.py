import re
import ast
import json
from config import RANDOM_SEED, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=GROQ_TEMPERATURE,
    max_tokens=GROQ_MAX_TOKENS,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# LLM with temperature=0 for structured extraction (more deterministic)
llm_extract = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.0,
    max_tokens=1024,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ── Dataset Registry ──────────────────────────────────────────────────────────
# Maps dataset names from papers to how we load them in Docker.
# If dataset not in registry → use domain default.
DATASET_REGISTRY = {
    # CV
    "mnist":        "mnist",
    "cifar-10":     "cifar10",
    "cifar10":      "cifar10",
    "cifar-100":    "cifar100",
    "cifar100":     "cifar100",
    "imagenet":     "mnist",        # too heavy → MNIST proxy
    # NLP
    "20newsgroups": "20newsgroups",
    "20news":       "20newsgroups",
    "imdb":         "20newsgroups", # proxy
    "sst":          "20newsgroups", # proxy
    "wmt14":        "20newsgroups", # too heavy → proxy
    "wmt":          "20newsgroups", # proxy
    # Recommendation
    "movielens":    "movielens",
    "movielens-100k": "movielens",
    "ml-100k":      "movielens",
    # RL
    "cartpole":     "cartpole",
    "atari":        "cartpole",     # proxy
    "mujoco":       "cartpole",     # proxy
}

# ── Training wrapper templates ────────────────────────────────────────────────
# These are WRAPPERS — they have a # MODEL_CLASS_HERE placeholder
# that gets replaced with Groq-generated model class.
# Fallback templates (old approach) used if Groq model generation fails.

ML_WRAPPER = f"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})

# MODEL_CLASS_HERE

# ── Data loading ──────────────────────────────────────────────────────────────
DATASET_LOADER_HERE

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = MODEL_NAME_HERE().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{{epoch+1}}/{{num_epochs}}] Loss: {{running_loss/len(train_loader):.4f}}")

model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Final Accuracy: {{accuracy:.2f}}%")
"""

NLP_WRAPPER = f"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})

print("Loading 20newsgroups dataset...")
data   = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), data_home='/app/data')
texts  = data.data
labels = data.target

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(texts).toarray().astype(np.float32)
y = np.array(labels, dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state={RANDOM_SEED})

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

num_classes = len(np.unique(y))
input_dim   = X_train.shape[1]

# MODEL_CLASS_HERE

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = MODEL_NAME_HERE(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
    print(f"Epoch [{{epoch+1}}/{{num_epochs}}] Loss: {{running_loss/len(train_loader):.4f}}")

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
print(f"Final Accuracy: {{accuracy:.2f}}%")
"""

RECOMMENDATION_WRAPPER = f"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import math
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})

print("Loading MovieLens 100K dataset...")
ratings = pd.read_csv('./data/ml-100k/u.data', sep='\\t',
                      names=['user_id', 'item_id', 'rating', 'timestamp'])
num_users = ratings['user_id'].max() + 1
num_items = ratings['item_id'].max() + 1
print(f"Users: {{num_users}}, Items: {{num_items}}")

user_ids    = torch.tensor(ratings['user_id'].values, dtype=torch.long)
item_ids    = torch.tensor(ratings['item_id'].values, dtype=torch.long)
rating_vals = torch.tensor(ratings['rating'].values,  dtype=torch.float32)

indices  = list(range(len(ratings)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state={RANDOM_SEED})

train_dataset = TensorDataset(user_ids[train_idx], item_ids[train_idx], rating_vals[train_idx])
test_dataset  = TensorDataset(user_ids[test_idx],  item_ids[test_idx],  rating_vals[test_idx])
train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=256, shuffle=False)

# MODEL_CLASS_HERE

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = MODEL_NAME_HERE(num_users, num_items).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
    print(f"Epoch [{{epoch+1}}/{{num_epochs}}] Loss: {{running_loss/len(train_loader):.4f}}")

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
print(f"RMSE: {{rmse:.4f}}")
print(f"Final Accuracy: {{accuracy:.2f}}%")
"""

RL_WRAPPER = f"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})
random.seed({RANDOM_SEED})

# MODEL_CLASS_HERE

env        = gym.make('CartPole-v1')
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model      = MODEL_NAME_HERE(state_dim, action_dim).to(device)
target_net = MODEL_NAME_HERE(state_dim, action_dim).to(device)
target_net.load_state_dict(model.state_dict())
optimizer  = optim.Adam(model.parameters(), lr=1e-3)
criterion  = nn.MSELoss()

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(args)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

buffer        = ReplayBuffer()
epsilon       = 1.0
epsilon_min   = 0.01
epsilon_decay = 0.995
gamma         = 0.99
batch_size    = 64
num_episodes  = 500
reward_history= []

for episode in range(num_episodes):
    result = env.reset()
    state  = result[0] if isinstance(result, tuple) else result
    total_reward = 0

    for _ in range(500):
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(s).argmax().item()

        result     = env.step(action)
        next_state, reward, done = result[0], result[1], result[2]
        buffer.push(state, action, reward, next_state, done)
        state        = next_state
        total_reward += reward

        if len(buffer) >= batch_size:
            batch       = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states      = torch.FloatTensor(np.array(states)).to(device)
            actions     = torch.LongTensor(actions).to(device)
            rewards_t   = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).to(device)
            dones_t     = torch.FloatTensor(dones).to(device)

            q_values      = model(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q_values = target_net(next_states).max(1)[0].detach()
            targets       = rewards_t + gamma * next_q_values * (1 - dones_t)

            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    reward_history.append(total_reward)

    if (episode + 1) % 50 == 0:
        avg = np.mean(reward_history[-50:])
        print(f"Episode {{episode+1}}/{{num_episodes}} | Avg Reward (last 50): {{avg:.2f}} | Epsilon: {{epsilon:.3f}}")

    if (episode + 1) % 10 == 0:
        target_net.load_state_dict(model.state_dict())

env.close()
avg_reward = np.mean(reward_history[-50:])
accuracy   = min(100.0, avg_reward / 5.0)
print(f"Final Accuracy: {{accuracy:.2f}}% (avg reward: {{avg_reward:.2f}})")
"""

GRAPH_WRAPPER = f"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})

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

# MODEL_CLASS_HERE

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = MODEL_NAME_HERE(num_features, num_classes).to(device)
X, y, adj = X.to(device), y.to(device), adj.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out  = model(X, adj)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {{epoch+1}}/200 | Loss: {{loss.item():.4f}}")

model.eval()
with torch.no_grad():
    out      = model(X, adj)
    preds    = out[test_mask].argmax(dim=1)
    correct  = (preds == y[test_mask]).sum().item()
    total    = test_mask.sum().item()
    accuracy = 100.0 * correct / total

print(f"Final Accuracy: {{accuracy:.2f}}%")
"""

ALGORITHM_TEMPLATE = f"""
import numpy as np
import random
random.seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})
"""

# ── Fallback model classes (used if Groq generation fails) ───────────────────
ML_FALLBACK_MODEL = """
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.LazyLinear(128)
        self.fc2   = nn.Linear(128, 10)
        self.drop  = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)
"""

NLP_FALLBACK_MODEL = """
class TextClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1  = nn.Linear(input_dim, 256)
        self.bn1  = nn.BatchNorm1d(256)
        self.fc2  = nn.Linear(256, 128)
        self.fc3  = nn.Linear(128, num_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)
"""

REC_FALLBACK_MODEL = """
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
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
"""

RL_FALLBACK_MODEL = """
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
"""

GRAPH_FALLBACK_MODEL = """
class GCN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GCN, self).__init__()
        self.fc1  = nn.Linear(input_dim, 64)
        self.fc2  = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x, adj):
        x = F.relu(self.fc1(adj @ x))
        x = self.drop(x)
        return self.fc2(adj @ x)
"""

FALLBACK_MODELS = {
    "ml":             (ML_FALLBACK_MODEL,  "Net"),
    "nlp":            (NLP_FALLBACK_MODEL, "TextClassifier"),
    "recommendation": (REC_FALLBACK_MODEL, "MatrixFactorization"),
    "rl":             (RL_FALLBACK_MODEL,  "DQN"),
    "graph":          (GRAPH_FALLBACK_MODEL, "GCN"),
}

DOMAIN_WRAPPERS = {
    "ml":             ML_WRAPPER,
    "nlp":            NLP_WRAPPER,
    "recommendation": RECOMMENDATION_WRAPPER,
    "rl":             RL_WRAPPER,
    "graph":          GRAPH_WRAPPER,
}

ML_DATASET_LOADERS = {
    "mnist": """transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=128, shuffle=False)""",

    "cifar10": """transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_dataset = datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=128, shuffle=False)""",

    "cifar100": """transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_dataset = datasets.CIFAR100('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=128, shuffle=False)""",
}


# ═════════════════════════════════════════════════════════════════════════════
# GROQ CALL 1 — Extract structured paper info
# ═════════════════════════════════════════════════════════════════════════════

def extract_paper_structure(filtered_text, domain, paper_id=None):
    """
    Call 1: Extract structured JSON from paper.
    Returns dict with task, dataset, model_name, architecture details.
    Falls back to safe defaults if extraction fails.
    Caches result to disk — skips Groq call on repeat runs.
    """
    # ── Cache check — skip Groq if already extracted ──────────────────────────
    if paper_id:
        cache_path = os.path.join(
            os.path.dirname(__file__), '..', 'tests', 'structure',
            f'{paper_id}_structure.json'
        )
        cache_path = os.path.normpath(cache_path)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                print(f"   ♻️  Reusing cached paper structure (saved {cached.get('_cached_at', 'previously')})")
                print(f"   📋 Task: {cached.get('task', 'unknown')}")
                print(f"   📦 Dataset: {cached.get('dataset', 'unknown')}")
                print(f"   🏗️  Model: {cached.get('model_name', 'unknown')}")
                return cached
            except Exception:
                pass  # cache corrupt — re-extract

    print("🔍 Extracting paper structure...")

    prompt = f"""You are a machine learning paper analyzer.

Extract structured information from the paper below and return ONLY valid JSON.
No explanation, no markdown, no extra text — ONLY the JSON object.

Required fields:
- task: what the paper is solving (e.g. "image classification", "machine translation")
- dataset: primary dataset used (e.g. "MNIST", "WMT14", "CIFAR-10")
- model_name: name of the model class to generate (e.g. "ResNet", "Transformer", "DQN") — valid Python class name
- metric: evaluation metric (e.g. "accuracy", "BLEU", "RMSE")
- architecture: dict with key architecture details relevant to domain:
  - for CV: {{"layers": 4, "filters": 64, "use_residual": true}}
  - for NLP: {{"layers": 6, "heads": 8, "hidden_size": 512, "use_attention": true}}
  - for RL: {{"hidden_size": 128, "use_target_network": true}}
  - for graph: {{"layers": 2, "hidden_size": 64, "use_attention": false}}
  - for recommendation: {{"embedding_dim": 50, "layers": 2}}
- hyperparams: dict with {{lr, batch_size, epochs, dropout, weight_decay}}

Domain: {domain}

PAPER:
{filtered_text[:6000]}

Return ONLY valid JSON:"""

    try:
        response    = llm_extract.invoke(prompt)
        raw         = response.content.strip()

        # Strip markdown if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        extracted = json.loads(raw.strip())

        # Validate required fields
        required = ["task", "dataset", "model_name", "metric", "architecture", "hyperparams"]
        missing  = [k for k in required if k not in extracted]
        if missing:
            print(f"⚠️  Missing fields: {missing} — using defaults")
            extracted = _fill_defaults(extracted, domain)

        # Sanitize model_name — must be valid Python class name
        model_name = extracted.get("model_name", "Model")
        model_name = re.sub(r'[^a-zA-Z0-9_]', '', model_name)
        if not model_name or model_name[0].isdigit():
            model_name = "Model"
        extracted["model_name"] = model_name

        print(f"   📋 Task: {extracted.get('task', 'unknown')}")
        print(f"   📦 Dataset: {extracted.get('dataset', 'unknown')}")
        print(f"   🏗️  Model: {extracted.get('model_name', 'unknown')}")

        # ── Save to cache ─────────────────────────────────────────────────────
        if paper_id:
            try:
                from datetime import datetime
                extracted['_cached_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                with open(cache_path, 'w') as f:
                    json.dump(extracted, f, indent=2)
                print(f"   💾 Structure cached to tests/structure/{paper_id}_structure.json")
            except Exception as e:
                print(f"   ⚠️  Cache save failed: {e}")

        return extracted

    except Exception as e:
        print(f"⚠️  Paper structure extraction failed ({e}) — using defaults")
        return _fill_defaults({}, domain)


def _fill_defaults(partial, domain):
    """Fill missing fields with safe domain defaults."""
    defaults = {
        "ml":             {"task": "image classification", "dataset": "MNIST",   "model_name": "Net",                "metric": "accuracy"},
        "nlp":            {"task": "text classification",  "dataset": "20newsgroups", "model_name": "TextClassifier","metric": "accuracy"},
        "recommendation": {"task": "rating prediction",    "dataset": "MovieLens","model_name": "MatrixFactorization","metric": "RMSE"},
        "rl":             {"task": "control",               "dataset": "CartPole", "model_name": "DQN",               "metric": "reward"},
        "graph":          {"task": "node classification",   "dataset": "synthetic","model_name": "GCN",               "metric": "accuracy"},
    }
    base = defaults.get(domain, defaults["ml"])
    for k, v in base.items():
        if k not in partial:
            partial[k] = v
    if "architecture" not in partial:
        partial["architecture"] = {}
    if "hyperparams" not in partial:
        partial["hyperparams"] = {}
    return partial


# ═════════════════════════════════════════════════════════════════════════════
# GROQ CALL 2 — Generate model class
# ═════════════════════════════════════════════════════════════════════════════

def generate_model_class(paper_info, domain, filtered_text):
    """
    Call 2: Generate PyTorch model class from extracted paper info.
    Returns (model_code, model_name) or None if generation fails.
    Retries up to 3 times with increasingly explicit prompts.
    """
    model_name = paper_info.get("model_name", "Model")
    arch       = paper_info.get("architecture", {})
    task       = paper_info.get("task", "")

    # Domain-specific constructor signature hints
    signature_hints = {
        "ml":             f"def __init__(self):",
        "nlp":            f"def __init__(self, input_dim, num_classes):",
        "recommendation": f"def __init__(self, num_users, num_items, embedding_dim=50):",
        "rl":             f"def __init__(self, state_dim, action_dim):",
        "graph":          f"def __init__(self, input_dim, num_classes):",
    }
    sig = signature_hints.get(domain, "def __init__(self):")

    # Architecture sanity checks to include in prompt
    arch_checks = {
        "ml":   (
            f"DATASET IS {paper_info.get('dataset', 'MNIST').upper()}. "
            f"Input channels: {'1 (grayscale)' if 'cifar' not in paper_info.get('dataset','mnist').lower() else '3 (RGB)'}. "
            f"First nn.Conv2d MUST use {'1' if 'cifar' not in paper_info.get('dataset','mnist').lower() else '3'} input channels. "
            "STRICT RULES — follow ALL of these exactly: "
            "1. Use MAXIMUM 2 stride=2 operations total — images are small. "
            "2. Use MAXIMUM 4 residual/conv blocks — CPU cannot handle 17+ blocks in reasonable time. "
            "3. NEVER hardcode a flatten size — use nn.AdaptiveAvgPool2d((1,1)) then flatten. "
            "4. Final output layer must output exactly 10 classes. "
            "5. Include residual/skip connections if the paper uses them — that is the key contribution. "
            "6. Each residual block: 2 conv layers with BatchNorm and ReLU, plus skip connection."
        ),
        "nlp":   (
            "Input is TF-IDF float vectors of shape (batch_size, input_dim). "
            "DO NOT use nn.Embedding — input is already float, not token indices. "
            "First layer MUST be nn.Linear(input_dim, hidden_size) to project input. "
            "If using nn.MultiheadAttention(hidden_size, num_heads, batch_first=True): "
            "  MUST unsqueeze before: x = x.unsqueeze(1)  # (batch, 1, hidden) "
            "  Call attention: out, _ = self.attn(x, x, x) "
            "  MUST squeeze after: x = out.squeeze(1)  # (batch, hidden) "
            "Final output: nn.Linear(hidden_size, num_classes). "
            "forward() input shape: (batch_size, input_dim). "
            "forward() output shape: (batch_size, num_classes). "
            "NEVER return 3D tensor from forward() — always squeeze to 2D before returning."
        ),
        "rl":    "Must include nn.Linear layers with ReLU activations. No nn.Embedding.",
        "graph": "Must accept (x, adj) in forward(). Use matrix multiplication for message passing.",
        "recommendation": "Must include nn.Embedding for users and items.",
    }
    arch_hint = arch_checks.get(domain, "")

    prompt = f"""You are an expert PyTorch engineer implementing a research paper.

Generate ONLY the PyTorch model class. No imports, no training loop, no data loading.

Paper info:
- Task: {task}
- Model: {model_name}
- Architecture details: {json.dumps(arch)}

STRICT REQUIREMENTS:
1. Class name must be exactly: {model_name}
2. Constructor signature must be: {sig}
3. Must have forward() method
4. {arch_hint}
5. Use only: nn.Linear, nn.Conv2d, nn.LSTM, nn.MultiheadAttention, nn.Embedding, nn.LayerNorm, nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, F.relu, F.softmax
6. No custom CUDA, no external libraries
7. Keep it runnable on CPU

Return ONLY the class definition, nothing else:

class {model_name}(nn.Module):
    {sig}
        ..."""

    for attempt in range(1, 4):
        try:
            print(f"🏗️  Generating model class (attempt {attempt}/3)...")
            response   = llm.invoke(prompt)
            model_code = response.content.strip()

            # Strip markdown
            if "```python" in model_code:
                model_code = model_code.split("```python")[1].split("```")[0]
            elif "```" in model_code:
                model_code = model_code.split("```")[1].split("```")[0]

            model_code = model_code.strip()

            # ── Static validation ─────────────────────────────────────────
            issues = _validate_model_class(model_code, model_name, domain)
            if issues:
                print(f"   ⚠️  Validation issues: {issues}")
                if attempt < 3:
                    # Make prompt more explicit on retry
                    prompt += f"\n\nPrevious attempt had issues: {issues}. Fix them."
                    continue

            print(f"   ✅ Model class generated: {model_name}")
            return model_code, model_name

        except Exception as e:
            print(f"   ❌ Attempt {attempt} failed: {e}")

    print("   ⚠️  All attempts failed — using fallback model")
    return None, None


def _validate_model_class(code, model_name, domain):
    """Static validation of generated model class. Returns list of issues."""
    issues = []

    # 1. Must be valid Python
    try:
        ast.parse(code)
    except SyntaxError as e:
        issues.append(f"SyntaxError: {e}")
        return issues  # no point checking further

    # 2. Must contain the class definition
    if f"class {model_name}" not in code:
        issues.append(f"Missing class {model_name}")

    # 3. Must contain forward()
    if "def forward(" not in code:
        issues.append("Missing forward() method")

    # 4. Domain-specific checks
    if domain == "ml" and "nn.Conv2d" not in code and "Conv2d" not in code:
        issues.append("CV model missing Conv2d")

    # ML channel check skipped — channel count depends on dataset (MNIST=1, CIFAR=3)
    # This is handled in the prompt instead

    if domain == "nlp" and "nn.Embedding" in code:
        issues.append("NLP model uses nn.Embedding but input is TF-IDF floats — use nn.Linear as first layer instead")

    if domain == "nlp" and "MultiheadAttention" in code and "unsqueeze" not in code:
        issues.append("NLP model uses MultiheadAttention but missing unsqueeze(1) — add x=x.unsqueeze(1) before attention and x=x.squeeze(1) after")

    if domain == "nlp" and "MultiheadAttention" in code:
        import re as _re
        # Check for 2-arg call pattern — missing value argument
        if _re.search(r'self\.\w+\(\s*\w+\s*,\s*\w+\s*\)', code) and not _re.search(r'self\.\w+\(\s*\w+\s*,\s*\w+\s*,\s*\w+\s*\)', code):
            issues.append("MultiheadAttention called with 2 args — must use 3: attn(x, x, x) for self-attention")
        # Check for missing batch_first=True
        if "MultiheadAttention(" in code and "batch_first=True" not in code:
            issues.append("MultiheadAttention missing batch_first=True — required for (batch, seq, dim) input")
        # Check for missing tuple unpack
        if _re.search(r'=\s*self\.\w+\(x,\s*x,\s*x\)', code) and "_ =" not in code and ", _" not in code:
            issues.append("MultiheadAttention output not unpacked as tuple: use 'out, _ = attn(x, x, x)'")

    if domain == "graph" and "adj" not in code:
        issues.append("Graph model missing adjacency matrix usage")

    if domain == "recommendation" and "Embedding" not in code:
        issues.append("Recommendation model missing Embedding layers")

    return issues


# ═════════════════════════════════════════════════════════════════════════════
# MAIN generate_code() — called by pipeline.py (same interface as before)
# ═════════════════════════════════════════════════════════════════════════════

def generate_code(filtered_text, domain="ml", paper_id=None):
    print("🤖 Sending to Groq for code generation...")

    # ── Algorithm domain: full Groq generation (unchanged) ───────────────────
    if domain == "algorithm":
        prompt = f"""You are an expert software engineer. Write a complete runnable Python script.

STRICT RULES:
- Implement the algorithm described in the context below
- Use only standard Python libraries
- Add clear comments explaining each step
- Include at least 3 test cases with print statements
- Always print: print(f"Final Accuracy: {{result:.2f}}%") or print(f"Algorithm Result: {{result}}")

ALGORITHM CONTEXT:
{filtered_text}

Write the complete Python implementation now:"""
        response    = llm.invoke(prompt)
        groq_output = response.content.strip()
        if "```python" in groq_output:
            groq_output = groq_output.split("```python")[1].split("```")[0]
        elif "```" in groq_output:
            groq_output = groq_output.split("```")[1].split("```")[0]
        print("✅ Code generated successfully!")
        return ALGORITHM_TEMPLATE + "\n" + groq_output

    # ── All other domains: two-call pipeline ─────────────────────────────────

    # Call 1: extract structured paper info
    paper_info = extract_paper_structure(filtered_text, domain, paper_id=paper_id)

    # Call 2: generate model class
    model_code, model_name = generate_model_class(paper_info, domain, filtered_text)

    # Determine if we're using Groq model or fallback
    using_fallback = False
    if model_code is None:
        print("   🔄 Using fallback template model")
        fallback       = FALLBACK_MODELS.get(domain, FALLBACK_MODELS["ml"])
        model_code     = fallback[0]
        model_name     = fallback[1]
        using_fallback = True

    # ── Assemble final code ───────────────────────────────────────────────────
    wrapper = DOMAIN_WRAPPERS.get(domain, ML_WRAPPER)

    # Inject model class
    code = wrapper.replace("# MODEL_CLASS_HERE", model_code)

    # Inject model name for instantiation
    code = code.replace("MODEL_NAME_HERE", model_name)

    # Inject dataset loader for ML domain
    if domain == "ml":
        dataset_key    = _resolve_dataset(paper_info.get("dataset", ""), domain)
        dataset_loader = ML_DATASET_LOADERS.get(dataset_key, ML_DATASET_LOADERS["mnist"])
        code           = code.replace("DATASET_LOADER_HERE", dataset_loader)

    # Patch hyperparams
    hyperparams = paper_info.get("hyperparams", {})
    code        = _patch_hyperparams(code, domain, hyperparams)

    # Safety fixes
    code = _safety_fixes(code, domain)

    if using_fallback:
        print("✅ Code generated (fallback model — paper architecture extraction failed)")
    else:
        print(f"✅ Code generated successfully! Model: {model_name}")

    return code


def _resolve_dataset(dataset_str, domain):
    """Map paper's dataset string to registry key."""
    ds = dataset_str.lower().strip()
    for key in DATASET_REGISTRY:
        if key in ds:
            return DATASET_REGISTRY[key]
    # Domain defaults
    defaults = {"ml": "mnist", "nlp": "20newsgroups", "recommendation": "movielens",
                "rl": "cartpole", "graph": "synthetic"}
    return defaults.get(domain, "mnist")


def _patch_hyperparams(code, domain, hyperparams):
    """Patch hyperparameters from extracted paper info."""
    lr           = _safe_float(hyperparams.get("lr"),           domain)
    batch_size   = _safe_int(hyperparams.get("batch_size"),     domain)
    epochs       = _safe_int(hyperparams.get("epochs"),         domain)
    dropout      = _safe_float(hyperparams.get("dropout"),      None) or 0.3
    weight_decay = _safe_float(hyperparams.get("weight_decay"), None) or 1e-4

    if domain == "ml":
        momentum = _safe_float(hyperparams.get("momentum"), None) or 0.9
        # Safety caps for MNIST training on CPU
        if momentum < 0.8:
            momentum = 0.9       # paper reports ImageNet momentum — default to 0.9 for MNIST
        if lr >= 0.1:
            lr = 0.01            # ImageNet lr (0.1) is too high for MNIST — cap at 0.01
        batch_size = max(batch_size, 64)   # minimum 64 — tiny batches are slow and noisy
        batch_size = min(batch_size, 256)  # maximum 256 — prevent OOM
        code = code.replace("lr=0.01",           f"lr={lr}")
        code = code.replace("momentum=0.9",      f"momentum={momentum}")
        code = code.replace("batch_size=128",    f"batch_size={batch_size}")
        code = code.replace("num_epochs = 5",    f"num_epochs = {epochs}")
        code = code.replace("weight_decay=1e-4", f"weight_decay={weight_decay}")
        print(f"📌 Patched ML params: lr={lr}, momentum={momentum}, batch_size={batch_size}, epochs={epochs}")

    elif domain == "nlp":
        # Cap batch_size — tiny batches cause slow training
        batch_size = max(batch_size, 64)
        batch_size = min(batch_size, 256)
        # Cap lr — Transformer paper uses warmup schedule, 0.001 is too high without it
        if lr > 1e-4:
            lr = 1e-4
        code = code.replace("lr=1e-3",         f"lr={lr}")
        code = code.replace("batch_size=64",   f"batch_size={batch_size}")
        code = code.replace("num_epochs = 10", f"num_epochs = {epochs}")
        print(f"📌 Patched NLP params: lr={lr}, batch_size={batch_size}, epochs={epochs}")

    elif domain == "recommendation":
        embedding_dim = _safe_int(hyperparams.get("embedding_dim"), None) or 50
        code = code.replace("lr=1e-3",          f"lr={lr}")
        code = code.replace("embedding_dim=50", f"embedding_dim={embedding_dim}")
        code = code.replace("num_epochs = 5",   f"num_epochs = {epochs}")
        code = code.replace("batch_size=256",   f"batch_size={batch_size}")
        print(f"📌 Patched Rec params: lr={lr}, embedding_dim={embedding_dim}, epochs={epochs}")

    elif domain == "rl":
        gamma         = _safe_float(hyperparams.get("gamma"), None) or 0.99
        epsilon_decay = _safe_float(hyperparams.get("epsilon_decay"), None) or 0.995
        episodes      = _safe_int(hyperparams.get("episodes"), None) or 500
        episodes      = min(episodes, 500)
        code = code.replace("lr=1e-3",             f"lr={lr}")
        code = code.replace("gamma         = 0.99",  f"gamma         = {gamma}")
        code = code.replace("epsilon_decay = 0.995", f"epsilon_decay = {epsilon_decay}")
        code = code.replace("num_episodes  = 500",   f"num_episodes  = {episodes}")
        print(f"📌 Patched RL params: lr={lr}, gamma={gamma}, episodes={episodes}")

    elif domain == "graph":
        code = code.replace("lr=0.01",           f"lr={lr}")
        code = code.replace("weight_decay=5e-4", f"weight_decay={weight_decay}")
        print(f"📌 Patched Graph params: lr={lr}, weight_decay={weight_decay}")

    return code


def _safe_float(val, domain):
    """Parse float, return domain-appropriate default if invalid."""
    defaults = {"ml": 0.01, "nlp": 1e-3, "recommendation": 1e-3, "rl": 1e-3, "graph": 0.01}
    try:
        v = float(val)
        if v <= 0 or v > 1.0:
            return defaults.get(domain, 0.01)
        return v
    except (TypeError, ValueError):
        return defaults.get(domain, 0.01)


def _safe_int(val, domain):
    """Parse int, return domain-appropriate default if invalid."""
    epoch_defaults  = {"ml": 5,  "nlp": 10, "recommendation": 5,  "rl": 500, "graph": 200}
    batch_defaults  = {"ml": 128,"nlp": 64, "recommendation": 256,"rl": 64,  "graph": 32}
    try:
        v = int(float(val))
        if v <= 0:
            return epoch_defaults.get(domain, 5)
        # Cap epochs
        epoch_caps = {"ml": 10, "nlp": 15, "recommendation": 10, "rl": 500, "graph": 200}
        if v > epoch_caps.get(domain, 10) and v < 1000:
            return min(v, epoch_caps.get(domain, 10))
        return v
    except (TypeError, ValueError):
        return epoch_defaults.get(domain, 5)


def _safety_fixes(code, domain):
    """Global safety post-processing."""
    if domain == "ml":
        # Fix hardcoded flatten
        code = re.sub(r'x\s*=\s*x\.view\(\s*-1\s*,\s*[^\)]+\)',
                      'x = x.view(x.size(0), -1)', code)
        # Fix large hardcoded Linear input sizes → LazyLinear
        code = re.sub(r'nn\.Linear\(\s*\d{4,}\s*,', 'nn.LazyLinear(', code)
        # Cap self.layers/self.num_layers to 4 — too many layers = hours on CPU
        code = re.sub(r'(self\.layers|self\.num_layers)\s*=\s*([5-9]|[1-9]\d+)', r'\1 = 4  # capped for CPU', code)
        # Cap self.layers to 4 — 10+ layers on CPU takes hours
        code = re.sub(r'self\.layers\s*=\s*([5-9]|\d{2,})', 'self.layers = 4', code)
        code = re.sub(r'self\.num_layers\s*=\s*([5-9]|\d{2,})', 'self.num_layers = 4', code)
        # Fix 3-channel Conv2d → 1-channel only when MNIST is loaded
        # CIFAR10/CIFAR100 are 3-channel RGB — don't patch those
        if "datasets.MNIST" in code and "CIFAR" not in code:
            code = re.sub(r'nn\.Conv2d\(\s*3\s*,', 'nn.Conv2d(1,', code)
        # Fix too many blocks — cap range() to 4 for CPU feasibility
        # e.g. range(17) → range(4), range(self.layers // 2) is harder to catch
        code = re.sub(r'range\(\s*(1[0-9]|[2-9]\d+)\s*\)', 'range(4)', code)
        code = re.sub(r'range\(self\.layers\s*//\s*\d+\)', 'range(4)', code)
        code = re.sub(r'range\(self\.num_layers\)', 'range(4)', code)
        # Add AdaptiveAvgPool before LazyLinear if not present
        # This prevents spatial dimension collapse on small images
        if "LazyLinear" in code and "AdaptiveAvgPool" not in code and "adaptive" not in code.lower():
            pool_line = "x = F.adaptive_avg_pool2d(x, (1, 1)) if x.dim() == 4 else x"
            flat_line = "x = x.view(x.size(0), -1)"
            code = code.replace(flat_line, pool_line + "\n        " + flat_line)

    # Fix common Groq capitalization typo — MultiHeadAttention → MultiheadAttention
    code = code.replace("nn.MultiHeadAttention", "nn.MultiheadAttention")

    # Remove deprecated torchtext
    if "import torchtext" in code or "from torchtext" in code:
        code = code.replace("import torchtext", "# torchtext deprecated")
        code = code.replace("from torchtext",   "# torchtext deprecated")

    # Fix common Groq capitalisation mistakes
    code = code.replace("nn.MultiHeadAttention", "nn.MultiheadAttention")
    code = code.replace("nn.Multiheadattention", "nn.MultiheadAttention")
    code = code.replace("nn.Multi_Head_Attention", "nn.MultiheadAttention")

    return code


# ── Keep old helpers for backward compatibility ───────────────────────────────
def fix_hardcoded_flatten(code):
    code = re.sub(r'x\s*=\s*x\.view\(\s*-1\s*,\s*[^\)]+\)',
                  'x = x.view(x.size(0), -1)', code)
    return code