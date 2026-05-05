import pickle
from typing import Dict, List

import numpy as np


def _stack(items: List[np.ndarray], dtype=np.float32) -> np.ndarray:
    return np.asarray(items, dtype=dtype)


def load_trajectory_dataset(
    trajectory_path: str,
    reward_is_negative_cost: bool = True,
) -> Dict[str, np.ndarray]:
    with open(trajectory_path, "rb") as f:
        trajectories = pickle.load(f)

    observations = []
    actions = []
    rewards = []
    next_observations = []
    next_actions = []
    terminals = []
    timeouts = []

    for traj in trajectories:
        states = traj["states"]
        traj_actions = traj["actions"]
        traj_rewards = traj["rewards"]

        traj_len = len(states)
        if traj_len == 0:
            continue

        for t in range(traj_len):
            obs = np.asarray(states[t], dtype=np.float32)
            action = np.asarray(traj_actions[t], dtype=np.float32)
            reward = float(traj_rewards[t])
            if reward_is_negative_cost:
                reward = -reward

            if t + 1 < traj_len:
                next_obs = np.asarray(states[t + 1], dtype=np.float32)
                next_action = np.asarray(traj_actions[t + 1], dtype=np.float32)
                terminal = 0.0
                timeout = 0.0
            else:
                next_obs = obs.copy()
                next_action = action.copy()
                terminal = 1.0
                timeout = 1.0

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            next_actions.append(next_action)
            terminals.append(terminal)
            timeouts.append(timeout)

    return {
        "observations": _stack(observations),
        "actions": _stack(actions),
        "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
        "next_observations": _stack(next_observations),
        "next_actions": _stack(next_actions),
        "terminals": np.asarray(terminals, dtype=np.float32).reshape(-1, 1),
        "timeouts": np.asarray(timeouts, dtype=np.float32).reshape(-1, 1),
    }
