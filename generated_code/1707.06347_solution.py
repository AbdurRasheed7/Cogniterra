import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import deque
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


import gymnasium as gym
import random
from collections import deque

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        self.hidden_size = 64
        self.use_target_network = True

        self.actor = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        if self.use_target_network:
            self.target_actor = nn.Sequential(
                nn.Linear(state_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, action_dim)
            )

            self.target_critic = nn.Sequential(
                nn.Linear(state_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1)
            )

            for param in self.target_actor.parameters():
                param.requires_grad = False

            for param in self.target_critic.parameters():
                param.requires_grad = False

    def forward(self, state):
        policy = self.actor(state)
        value = self.critic(state)
        return policy, value

env        = gym.make('CartPole-v1')
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model      = PPO(state_dim, action_dim).to(device)
target_net = PPO(state_dim, action_dim).to(device)
target_net.load_state_dict(model.state_dict())
optimizer  = optim.Adam(model.parameters(), lr=0.001)
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
epsilon_decay = 0.01
gamma         = 0.99
batch_size    = 64
num_episodes  = 5
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
                policy, _ = model(s)
                action = torch.argmax(policy).item()

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

            policy, q_values = model(states)
            q_values      = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            _, next_q_values = target_net(next_states)
            next_q_values = next_q_values.squeeze().max(0)[0].detach()
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
        print(f"Episode {episode+1}/{num_episodes} | Avg Reward (last 50): {avg:.2f} | Epsilon: {epsilon:.3f}")

    if (episode + 1) % 10 == 0:
        target_net.load_state_dict(model.state_dict())

env.close()
avg_reward = np.mean(reward_history[-50:])
accuracy   = min(100.0, avg_reward / 5.0)
print(f"Final Accuracy: {accuracy:.2f}% (avg reward: {avg_reward:.2f})")
