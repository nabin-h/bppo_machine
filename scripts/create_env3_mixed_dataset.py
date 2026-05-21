import argparse
import json
import pickle
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from custom_env import DEFAULT_ENV_NAME, random_viable_action, sample_initial_observation, take_step, threshold_policy


def parse_thresholds(raw: str):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def compute_rtg(rewards, gamma: float):
    running = 0.0
    values = np.zeros(len(rewards), dtype=np.float64)
    for idx in reversed(range(len(rewards))):
        running = float(rewards[idx]) + float(gamma) * running
        values[idx] = running
    return values


def build_mix_tag(thresholds, random_fraction: float):
    thr_tag = "_".join(f"t{threshold}" for threshold in thresholds)
    rnd_tag = f"rnd{int(round(random_fraction * 100))}"
    return f"mix_{thr_tag}_equal_{rnd_tag}"


def build_output_filename(num_machines: int, mix_tag: str, num_episodes: int, rank_method: int):
    return f"traj_mixthr_MM{num_machines}_{mix_tag}_{num_episodes}_minR{rank_method}.pkl"


def choose_behavior_source(rng: random.Random, thresholds, random_fraction: float):
    threshold_share = (1.0 - random_fraction) / float(len(thresholds))
    cutoffs = [random_fraction]
    running = random_fraction
    for _ in thresholds:
        running += threshold_share
        cutoffs.append(running)

    sample = rng.random()
    if sample < cutoffs[0]:
        return ("random", None)

    for idx, threshold in enumerate(thresholds):
        if sample < cutoffs[idx + 1]:
            return (f"t{threshold}", threshold)
    return (f"t{thresholds[-1]}", thresholds[-1])


def create_mixed_dataset(num_machines: int, num_episodes: int, sim_timesteps: int, store_timesteps: int, thresholds, random_fraction: float, output_root: Path, gamma: float = 0.95, rank_method: int = 2, seed: int = 42, env_name: str = DEFAULT_ENV_NAME):
    if store_timesteps > sim_timesteps:
        store_timesteps = sim_timesteps

    mix_tag = build_mix_tag(thresholds, random_fraction)
    dataset_dir = output_root / f"MM{num_machines}" / mix_tag
    dataset_dir.mkdir(parents=True, exist_ok=True)

    chooser_rng = random.Random(seed)
    env_rng = np.random.RandomState(seed)
    random.seed(seed)
    np.random.seed(seed)

    all_trajectories = []
    behavior_counter = Counter()

    for traj_id in range(num_episodes):
        source_label, threshold = choose_behavior_source(chooser_rng, thresholds, random_fraction)
        behavior_counter[source_label] += 1
        states = sample_initial_observation(num_machines=num_machines, rng=env_rng, env_name=env_name).tolist()
        full_states, full_actions, full_rewards = [], [], []

        for _ in range(sim_timesteps):
            if threshold is None:
                actions = random_viable_action(states, rng=env_rng, env_name=env_name).tolist()
            else:
                actions = threshold_policy(states, threshold=threshold, env_name=env_name).tolist()
            next_states, cost = take_step(states, actions, rng=env_rng, env_name=env_name)
            full_states.append(states.copy())
            full_actions.append(list(actions))
            full_rewards.append(float(cost))
            states = next_states.tolist() if hasattr(next_states, 'tolist') else list(next_states)

        rtg_full = compute_rtg(full_rewards, gamma)
        all_trajectories.append({
            "index": [traj_id + 1] * min(store_timesteps, len(full_rewards)),
            "states": full_states[:store_timesteps],
            "actions": full_actions[:store_timesteps],
            "rewards": full_rewards[:store_timesteps],
            "rtg": rtg_full[:store_timesteps].tolist(),
            "behavior_source": source_label,
        })

    if rank_method == 1:
        all_trajectories.sort(key=lambda traj: traj["rtg"][0] if len(traj["rtg"]) > 0 else -np.inf, reverse=False)
    elif rank_method == 2:
        all_trajectories.sort(key=lambda traj: max(traj["rtg"]) if len(traj["rtg"]) > 0 else -np.inf, reverse=False)

    dataset_tag = mix_tag
    dataset_path = dataset_dir / build_output_filename(num_machines, mix_tag, num_episodes, rank_method)
    with open(dataset_path, "wb") as f:
        pickle.dump(all_trajectories, f)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "num_machines": int(num_machines),
        "num_episodes": int(num_episodes),
        "sim_timesteps": int(sim_timesteps),
        "store_timesteps": int(store_timesteps),
        "optimal_fraction": 1.0,
        "dataset_type": "threshold_mix",
        "dataset_tag": dataset_tag,
        "thresholds": [int(item) for item in thresholds],
        "random_fraction": float(random_fraction),
        "threshold_fraction_each": float((1.0 - random_fraction) / float(len(thresholds))),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "gamma": float(gamma),
        "rank_method": int(rank_method),
        "seed": int(seed),
        "env_name": str(env_name),
        "mix_tag": mix_tag,
        "behavior_counts": dict(behavior_counter),
    }

    metadata_path = dataset_dir / "dataset_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    manifest_path = output_root / f"dataset_manifest_MM{num_machines}_{mix_tag}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([metadata], f, indent=2)

    print(json.dumps({"dataset_path": str(dataset_path), "metadata_path": str(metadata_path), "manifest_path": str(manifest_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Create an env3 mixed dataset with random and threshold-policy trajectories.")
    parser.add_argument("--num_machines", type=int, required=True)
    parser.add_argument("--num_episodes", type=int, default=20000)
    parser.add_argument("--sim_timesteps", type=int, default=100)
    parser.add_argument("--store_timesteps", type=int, default=100)
    parser.add_argument("--thresholds", type=str, default="5,6,7")
    parser.add_argument("--random_fraction", type=float, default=0.20)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--rank_method", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_name", type=str, default="env3")
    args = parser.parse_args()

    create_mixed_dataset(args.num_machines, args.num_episodes, args.sim_timesteps, args.store_timesteps, parse_thresholds(args.thresholds), args.random_fraction, Path(args.output_root), args.gamma, args.rank_method, args.seed, args.env_name)


if __name__ == "__main__":
    main()
