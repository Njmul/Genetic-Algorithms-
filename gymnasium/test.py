import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from collections import deque, Counter


# Must match the improved training network (128, 128)
class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def select_action_greedy(model, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)
    return torch.argmax(q_values, dim=1).item()


def smooth_action(raw_action, action_history: deque):
    """
    Smooth action by taking the most frequent action in the recent window.
    Reduces jitter (rapid switching).
    """
    action_history.append(raw_action)
    return Counter(action_history).most_common(1)[0][0]


def evaluate_runs(
    checkpoint_path="checkpoint_best.pth",
    seed=123,
    n_runs=1000,
    render=False,
    smoothing_window=5,
    action_repeat=2,
    success_reward_threshold=200.0,
    print_each_streak=True,
    print_every=100,
):
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = Network(state_size, action_size)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    scores = []

    # ---- streak tracking ----
    current_streak = 0
    max_streak = 0
    streaks = []
    streak_start_ep = None
    successes = 0

    action_history = deque(maxlen=smoothing_window)

    for episode in range(1, n_runs + 1):

        # Important: seed only once, then randomize afterwards
        if episode == 1:
            state, _ = env.reset(seed=seed)
        else:
            state, _ = env.reset()

        terminated = truncated = False
        total_reward = 0.0
        action_history.clear()

        while not (terminated or truncated):
            raw_action = select_action_greedy(model, state)
            action = smooth_action(raw_action, action_history)

            # action repeat makes motion smoother
            for _ in range(action_repeat):
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

        scores.append(total_reward)

        # ---- success decision ----
        is_success = total_reward >= success_reward_threshold

        if is_success:
            successes += 1
            if current_streak == 0:
                streak_start_ep = episode
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            if current_streak > 0:
                streaks.append(current_streak)
                if print_each_streak:
                    print(
                        f"Success streak ended: length={current_streak} "
                        f"(episodes {streak_start_ep}–{episode-1})"
                    )
            current_streak = 0
            streak_start_ep = None

        if print_every and (episode % print_every == 0):
            print(
                f"Episode {episode:4d} | Reward: {total_reward:8.2f} | "
                f"Current streak: {current_streak:3d} | Max streak: {max_streak:3d}"
            )

    # close last streak if ended on success
    if current_streak > 0:
        streaks.append(current_streak)
        if print_each_streak:
            print(
                f"Success streak ended: length={current_streak} "
                f"(episodes {streak_start_ep}–{n_runs})"
            )

    env.close()

    print("\n===== Final Statistics =====")
    print(f"Checkpoint used              : {checkpoint_path}")
    print(f"Initial seed                 : {seed}")
    print(f"Total runs                   : {n_runs}")
    print(f"Mean reward                  : {np.mean(scores):.2f}")
    print(f"Std deviation                : {np.std(scores):.2f}")
    print(f"Min reward                   : {np.min(scores):.2f}")
    print(f"Max reward                   : {np.max(scores):.2f}")
    print(f"Success threshold (reward)   : {success_reward_threshold}")
    print(f"Total successes              : {successes}/{n_runs}")
    print(f"Maximum consecutive successes: {max_streak}")
    print(f"All success streaks (lengths): {streaks}")

    return scores, streaks, max_streak


if __name__ == "__main__":
    evaluate_runs(
        checkpoint_path="checkpoint_best.pth",
        seed=123,
        n_runs=1000,
        render=False,
        smoothing_window=5,
        action_repeat=2,
        success_reward_threshold=200.0,
        print_each_streak=True,
        print_every=100,
    )
