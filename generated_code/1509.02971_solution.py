
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class DDPG(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPG, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.output_activation = nn.Tanh()

    def forward(self, x):
        x = self.actor(x)
        return self.output_activation(x) * 2.0

env         = gym.make('Pendulum-v1')
state_dim   = env.observation_space.shape[0]   # 3
action_dim  = env.action_space.shape[0]         # 1 (continuous)
action_high = float(env.action_space.high[0])   # 2.0
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model     = DDPG(state_dim, action_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

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
gamma         = 0.01
batch_size    = 64
num_episodes  = 5
reward_history = []

for episode in range(num_episodes):
    result = env.reset()
    state  = result[0] if isinstance(result, tuple) else result
    total_reward = 0

    for _ in range(200):
        with torch.no_grad():
            s   = torch.FloatTensor(state).unsqueeze(0).to(device)
            out = model(s)
            # Handle tuple output (actor-critic) — use actor output (first element)
            if isinstance(out, tuple):
                out = out[0]
            # Continuous action — clamp to env action bounds
            action = out.squeeze().cpu().numpy()
            if action.ndim == 0:
                action = np.array([float(action)])
            action = np.clip(action, -action_high, action_high)

        result     = env.step(action)
        next_state, reward, done = result[0], result[1], result[2]
        buffer.push(state, action, reward, next_state, done)
        state        = next_state
        total_reward += reward

        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states_t      = torch.FloatTensor(np.array(states)).to(device)
            actions_t     = torch.FloatTensor(np.array(actions)).to(device)
            rewards_t     = torch.FloatTensor(rewards).to(device)
            next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
            dones_t       = torch.FloatTensor(dones).to(device)

            with torch.no_grad():
                next_out = model(next_states_t)
                if isinstance(next_out, tuple):
                    next_out = next_out[0]
                # TD target: use negative squared action as proxy value signal
                next_val = rewards_t + gamma * (-next_out.pow(2).mean(dim=-1)) * (1 - dones_t)

            curr_out = model(states_t)
            if isinstance(curr_out, tuple):
                curr_out = curr_out[0]
            curr_val = -curr_out.pow(2).mean(dim=-1)

            loss = criterion(curr_val, next_val.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    reward_history.append(total_reward)
    if (episode + 1) % 50 == 0:
        avg = np.mean(reward_history[-50:])
        print(f"Episode {episode+1}/{num_episodes} | Avg Reward (last 50): {avg:.2f}")

env.close()
avg_reward = np.mean(reward_history[-50:])
# Pendulum reward range: -1200 (worst) to -100 (good) → normalise to 0-100
accuracy   = min(100.0, max(0.0, (avg_reward + 1200) / 11.0))
print(f"Final Accuracy: {accuracy:.2f}% (avg reward: {avg_reward:.2f})")
