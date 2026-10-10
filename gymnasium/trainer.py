import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym


# -------------------------
# Reproducibility (optional)
# -------------------------
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -------------------------
# Hyperparameters (tuned)
# -------------------------
learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99
replay_buffer_size = int(2e5)   # a bit larger
tau = 1e-3                      # soft update
update_every = 4
learn_starts = 2000             # wait before learning
max_grad_norm = 10.0

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.9995          # slower decay than 0.995 (important)



# Environment

env = gym.make("LunarLander-v3")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("State size:", state_size, "| Actions:", action_size)



# Q-Network (slightly bigger)

class Network(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



# Replay Buffer

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states      = torch.from_numpy(np.vstack([b[0] for b in batch])).float().to(device)
        actions     = torch.from_numpy(np.vstack([b[1] for b in batch])).long().to(device)
        rewards     = torch.from_numpy(np.vstack([b[2] for b in batch])).float().to(device)
        next_states = torch.from_numpy(np.vstack([b[3] for b in batch])).float().to(device)
        dones       = torch.from_numpy(np.vstack([b[4] for b in batch]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)



# Agent (Double DQN + soft target)

class Agent:
    def __init__(self, state_size, action_size):
        self.q_local = Network(state_size, action_size).to(device)
        self.q_target = Network(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0
        self.total_steps = 0

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(action_size)

        s = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_local.eval()
        with torch.no_grad():
            q = self.q_local(s)
        self.q_local.train()
        return int(torch.argmax(q, dim=1).item())

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self.t_step = (self.t_step + 1) % update_every

        if self.total_steps < learn_starts:
            return

        if self.t_step == 0 and len(self.memory) >= minibatch_size:
            experiences = self.memory.sample(minibatch_size)
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences


        # action selection from local network
        next_actions = torch.argmax(self.q_local(next_states), dim=1, keepdim=True)

        # action evaluation from target network
        next_q = self.q_target(next_states).gather(1, next_actions).detach()

        q_targets = rewards + (discount_factor * next_q * (1 - dones))
        q_expected = self.q_local(states).gather(1, actions)

        # Huber loss (more stable than MSE)
        loss = F.smooth_l1_loss(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_local.parameters(), max_grad_norm)
        self.optimizer.step()

        # soft update
        self.soft_update(self.q_local, self.q_target, tau)

    @staticmethod
    def soft_update(local_model, target_model, tau_):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau_ * local_param.data + (1.0 - tau_) * target_param.data)



# Train

agent = Agent(state_size, action_size)

number_episodes = 10000
max_steps_per_episode = 1000
scores_window = deque(maxlen=100)

best_avg = -1e9
epsilon = epsilon_start

for ep in range(1, number_episodes + 1):
    state, _ = env.reset(seed=None)  # random starts
    score = 0.0

    for t in range(max_steps_per_episode):
        action = agent.act(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # 

        agent.step(state, action, reward, next_state, done)

        state = next_state
        score += reward
        if done:
            break

    scores_window.append(score)
    avg_score = np.mean(scores_window)

    # decay epsilon slowly
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # print progress
    print(f"\rEpisode {ep:4d} | Avg(100): {avg_score:7.2f} | Eps: {epsilon:.3f}", end="")
    if ep % 100 == 0:
        print(f"\rEpisode {ep:4d} | Avg(100): {avg_score:7.2f} | Eps: {epsilon:.3f}")

    # save best checkpoint (helps streak later)
    if avg_score > best_avg:
        best_avg = avg_score
        torch.save(agent.q_local.state_dict(), "checkpoint_best.pth")

    # solved condition
    if avg_score >= 200.0 and ep >= 100:
        print(f"\n Solved in {ep-100} episodes | Avg(100): {avg_score:.2f}")
        torch.save(agent.q_local.state_dict(), "checkpoint.pth")
        break

env.close()

print("\nTraining finished.")
print("Best avg(100) achieved:", best_avg)
print("Saved: checkpoint_best.pth (best), checkpoint.pth (solved if reached)")
