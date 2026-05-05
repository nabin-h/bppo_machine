import pickle
from typing import Any, Callable, Dict, Optional

import gym
import numpy as np

from custom_env import DEFAULT_ENV_NAME, project_action, split_observation


def load_reference_policy(policy_path: str):
    with open(policy_path, 'rb') as f:
        data = pickle.load(f)
    return data


def reference_action_for_state(reference_policy, state_key, action_dim: int) -> np.ndarray:
    if reference_policy is None:
        return np.zeros(action_dim, dtype=np.int32)
    policy_type = reference_policy.get('policy_type', 'table') if isinstance(reference_policy, dict) else 'table'
    if policy_type == 'component_threshold':
        threshold = int(reference_policy.get('threshold', 2))
        component_state_key = tuple(state_key[:action_dim])
        return np.asarray([1 if int(s) >= threshold else 0 for s in component_state_key], dtype=np.int32)
    policy_table = reference_policy.get('policy', {}) if isinstance(reference_policy, dict) else reference_policy
    if state_key in policy_table:
        return np.asarray(policy_table.get(state_key, (0,) * action_dim), dtype=np.int32)
    component_state_key = tuple(state_key[:action_dim])
    return np.asarray(policy_table.get(component_state_key, (0,) * action_dim), dtype=np.int32)


def _reference_suffix(label: str, policy) -> Optional[str]:
    if label == "__main__":
        if isinstance(policy, dict) and policy.get("policy_type") == "component_threshold":
            return f"t{int(policy.get('threshold', 2))}"
        return None
    if isinstance(policy, dict) and policy.get("policy_type") == "component_threshold":
        return f"t{int(policy.get('threshold', 2))}"
    return label


def _summarize_stats(stats: Dict[str, list]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key, values in stats.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = float(np.mean(arr))
        summary[f'{key}_std'] = float(np.std(arr))
    return summary


def _add_match_stats(summary: Dict[str, float],
                     projected_full_matches: list,
                     projected_match_dist_acc,
                     raw_full_matches: list,
                     raw_match_dist_acc,
                     num_episodes: int) -> None:
    if projected_full_matches:
        projected_full_matches_arr = np.asarray(
            projected_full_matches, dtype=np.float64)
        projected_mean = float(np.mean(projected_full_matches_arr))
        projected_std = float(np.std(projected_full_matches_arr))
        summary['full_match_pct'] = projected_mean
        summary['full_match_pct_std'] = projected_std
        summary['projected_full_match_pct'] = projected_mean
        summary['projected_full_match_pct_std'] = projected_std
        summary['match_distribution'] = (
            projected_match_dist_acc / num_episodes).tolist()
        summary['projected_match_distribution'] = summary['match_distribution']

    if raw_full_matches:
        raw_full_matches_arr = np.asarray(raw_full_matches, dtype=np.float64)
        summary['raw_full_match_pct'] = float(np.mean(raw_full_matches_arr))
        summary['raw_full_match_pct_std'] = float(np.std(raw_full_matches_arr))
        summary['raw_match_distribution'] = (
            raw_match_dist_acc / num_episodes).tolist()


def evaluate(agent: Any, env: gym.Env,
             num_episodes: int,
             discount: float = 0.95,
             reference_policy: Optional[Dict[tuple, tuple]] = None,
             extra_reference_policies: Optional[Dict[str, Dict[tuple, tuple]]] = None,
             env_name: str = DEFAULT_ENV_NAME) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    action_sum = []
    action_count = 0
    all_reference_policies = {}
    if reference_policy is not None:
        all_reference_policies["__main__"] = reference_policy
    if extra_reference_policies:
        all_reference_policies.update(extra_reference_policies)
    match_trackers = {}
    for label in all_reference_policies:
        match_trackers[label] = {
            "projected_full_matches": [],
            "raw_full_matches": [],
            "projected_match_dist_acc": None,
            "raw_match_dist_acc": None,
        }

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        t = 0
        discounted_return = 0.0
        discounted_cost = 0.0
        episode_corrections = 0
        projected_full_match_count = {}
        raw_full_match_count = {}
        episode_steps = 0
        action_dim = int(np.asarray(env.action_space.sample()).shape[-1])
        episode_projected_match_dist = {}
        episode_raw_match_dist = {}
        for label in all_reference_policies:
            projected_full_match_count[label] = 0
            raw_full_match_count[label] = 0
            episode_projected_match_dist[label] = np.zeros(
                action_dim + 1, dtype=np.float64)
            episode_raw_match_dist[label] = np.zeros(
                action_dim + 1, dtype=np.float64)
            if match_trackers[label]["projected_match_dist_acc"] is None:
                match_trackers[label]["projected_match_dist_acc"] = np.zeros(
                    action_dim + 1, dtype=np.float64)
                match_trackers[label]["raw_match_dist_acc"] = np.zeros(
                    action_dim + 1, dtype=np.float64)

        while not done:
            state_for_match = np.asarray(observation, dtype=np.int32)
            raw_action = agent.sample_actions(
                observation, temperature=0.0)  # 0.0 greedy action
            raw_action_arr = np.asarray(raw_action, dtype=np.int32)
            action_arr = project_action(observation, raw_action_arr, env_name=env_name).astype(np.float32)
            episode_corrections += int(np.sum(raw_action_arr != action_arr.astype(np.int32)))
            action_sum.append(action_arr)
            action_count += action_arr.size
            observation, reward, done, info = env.step(action_arr)
            discounted_return += (discount ** t) * float(reward)
            discounted_cost += (discount ** t) * float(-reward)
            state_key = tuple(state_for_match.tolist())
            for label, policy in all_reference_policies.items():
                optimal_action = reference_action_for_state(
                    policy, state_key, action_arr.shape[-1])
                raw_correct_actions = int(np.sum(raw_action_arr == optimal_action))
                projected_correct_actions = int(
                    np.sum(action_arr.astype(np.int32) == optimal_action))
                episode_raw_match_dist[label][raw_correct_actions] += 1
                episode_projected_match_dist[label][projected_correct_actions] += 1
                if raw_correct_actions == action_arr.shape[-1]:
                    raw_full_match_count[label] += 1
                if projected_correct_actions == action_arr.shape[-1]:
                    projected_full_match_count[label] += 1
            episode_steps += 1
            t += 1

        for k in info['episode'].keys():
            stats.setdefault(k, [])
            stats[k].append(info['episode'][k])
        stats.setdefault('discounted_return', [])
        stats['discounted_return'].append(discounted_return)
        stats.setdefault('discounted_cost', [])
        stats['discounted_cost'].append(discounted_cost)
        stats.setdefault('corrections', [])
        stats['corrections'].append(float(episode_corrections))
        stats.setdefault('raw_invalid_action_count', [])
        stats['raw_invalid_action_count'].append(float(episode_corrections))
        stats.setdefault('raw_invalid_action_rate', [])
        total_action_decisions = episode_steps * action_dim
        invalid_action_rate = (
            float(episode_corrections) / total_action_decisions
            if total_action_decisions > 0 else np.nan)
        stats['raw_invalid_action_rate'].append(invalid_action_rate)
        if episode_steps > 0:
            for label in all_reference_policies:
                match_trackers[label]["raw_full_matches"].append(
                    (raw_full_match_count[label] / episode_steps) * 100.0)
                match_trackers[label]["projected_full_matches"].append(
                    (projected_full_match_count[label] / episode_steps) * 100.0)
                match_trackers[label]["raw_match_dist_acc"] += episode_raw_match_dist[label]
                match_trackers[label]["projected_match_dist_acc"] += episode_projected_match_dist[label]

    summary = _summarize_stats(stats)

    if action_sum:
        stacked_actions = np.stack(action_sum)
        per_step_ones = np.sum(stacked_actions, axis=-1)
        summary['action_ones_per_step'] = float(np.mean(per_step_ones))
        summary['action_ones_per_step_std'] = float(np.std(per_step_ones))
        activation = np.mean(stacked_actions, axis=-1)
        summary['action_activation_rate'] = float(np.mean(activation))
        summary['action_activation_rate_std'] = float(np.std(activation))
    for label, policy in all_reference_policies.items():
        tmp = {}
        _add_match_stats(
            tmp,
            match_trackers[label]["projected_full_matches"],
            match_trackers[label]["projected_match_dist_acc"],
            match_trackers[label]["raw_full_matches"],
            match_trackers[label]["raw_match_dist_acc"],
            num_episodes,
        )
        suffix = _reference_suffix(label, policy)
        if label == "__main__":
            summary.update(tmp)
            if suffix:
                for key, value in tmp.items():
                    if key.endswith("_std"):
                        base = key[:-4]
                        summary[f"{base}_{suffix}_std"] = value
                    else:
                        summary[f"{key}_{suffix}"] = value
        elif suffix:
            for key, value in tmp.items():
                if key.endswith("_std"):
                    base = key[:-4]
                    summary[f"{base}_{suffix}_std"] = value
                else:
                    summary[f"{key}_{suffix}"] = value

    return summary


def evaluate_policy_fn(policy_fn: Callable[[np.ndarray], np.ndarray],
                       env: gym.Env,
                       num_episodes: int,
                       discount: float = 0.95,
                       reference_policy: Optional[Dict[tuple, tuple]] = None,
                       extra_reference_policies: Optional[Dict[str, Dict[tuple, tuple]]] = None,
                       env_name: str = DEFAULT_ENV_NAME) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    action_sum = []
    action_count = 0
    all_reference_policies = {}
    if reference_policy is not None:
        all_reference_policies["__main__"] = reference_policy
    if extra_reference_policies:
        all_reference_policies.update(extra_reference_policies)
    match_trackers = {}
    for label in all_reference_policies:
        match_trackers[label] = {
            "projected_full_matches": [],
            "raw_full_matches": [],
            "projected_match_dist_acc": None,
            "raw_match_dist_acc": None,
        }

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        t = 0
        discounted_return = 0.0
        discounted_cost = 0.0
        episode_corrections = 0
        projected_full_match_count = {}
        raw_full_match_count = {}
        episode_steps = 0
        action_dim = int(np.asarray(env.action_space.sample()).shape[-1])
        episode_projected_match_dist = {}
        episode_raw_match_dist = {}
        for label in all_reference_policies:
            projected_full_match_count[label] = 0
            raw_full_match_count[label] = 0
            episode_projected_match_dist[label] = np.zeros(
                action_dim + 1, dtype=np.float64)
            episode_raw_match_dist[label] = np.zeros(
                action_dim + 1, dtype=np.float64)
            if match_trackers[label]["projected_match_dist_acc"] is None:
                match_trackers[label]["projected_match_dist_acc"] = np.zeros(
                    action_dim + 1, dtype=np.float64)
                match_trackers[label]["raw_match_dist_acc"] = np.zeros(
                    action_dim + 1, dtype=np.float64)

        while not done:
            state_for_match = np.asarray(observation, dtype=np.int32)
            raw_action = policy_fn(observation)
            raw_action_arr = np.asarray(raw_action, dtype=np.int32)
            action_arr = project_action(observation, raw_action_arr, env_name=env_name).astype(np.float32)
            episode_corrections += int(np.sum(raw_action_arr != action_arr.astype(np.int32)))
            action_sum.append(action_arr)
            action_count += action_arr.size
            observation, reward, done, info = env.step(action_arr)
            discounted_return += (discount ** t) * float(reward)
            discounted_cost += (discount ** t) * float(-reward)
            state_key = tuple(state_for_match.tolist())
            for label, policy in all_reference_policies.items():
                optimal_action = reference_action_for_state(
                    policy, state_key, action_arr.shape[-1])
                raw_correct_actions = int(np.sum(raw_action_arr == optimal_action))
                projected_correct_actions = int(
                    np.sum(action_arr.astype(np.int32) == optimal_action))
                episode_raw_match_dist[label][raw_correct_actions] += 1
                episode_projected_match_dist[label][projected_correct_actions] += 1
                if raw_correct_actions == action_arr.shape[-1]:
                    raw_full_match_count[label] += 1
                if projected_correct_actions == action_arr.shape[-1]:
                    projected_full_match_count[label] += 1
            episode_steps += 1
            t += 1

        for k in info['episode'].keys():
            stats.setdefault(k, [])
            stats[k].append(info['episode'][k])
        stats.setdefault('discounted_return', [])
        stats['discounted_return'].append(discounted_return)
        stats.setdefault('discounted_cost', [])
        stats['discounted_cost'].append(discounted_cost)
        stats.setdefault('corrections', [])
        stats['corrections'].append(float(episode_corrections))
        stats.setdefault('raw_invalid_action_count', [])
        stats['raw_invalid_action_count'].append(float(episode_corrections))
        stats.setdefault('raw_invalid_action_rate', [])
        total_action_decisions = episode_steps * action_dim
        invalid_action_rate = (
            float(episode_corrections) / total_action_decisions
            if total_action_decisions > 0 else np.nan)
        stats['raw_invalid_action_rate'].append(invalid_action_rate)
        if episode_steps > 0:
            for label in all_reference_policies:
                match_trackers[label]["raw_full_matches"].append(
                    (raw_full_match_count[label] / episode_steps) * 100.0)
                match_trackers[label]["projected_full_matches"].append(
                    (projected_full_match_count[label] / episode_steps) * 100.0)
                match_trackers[label]["raw_match_dist_acc"] += episode_raw_match_dist[label]
                match_trackers[label]["projected_match_dist_acc"] += episode_projected_match_dist[label]

    summary = _summarize_stats(stats)

    if action_sum:
        stacked_actions = np.stack(action_sum)
        per_step_ones = np.sum(stacked_actions, axis=-1)
        summary['action_ones_per_step'] = float(np.mean(per_step_ones))
        summary['action_ones_per_step_std'] = float(np.std(per_step_ones))
        activation = np.mean(stacked_actions, axis=-1)
        summary['action_activation_rate'] = float(np.mean(activation))
        summary['action_activation_rate_std'] = float(np.std(activation))
    for label, policy in all_reference_policies.items():
        tmp = {}
        _add_match_stats(
            tmp,
            match_trackers[label]["projected_full_matches"],
            match_trackers[label]["projected_match_dist_acc"],
            match_trackers[label]["raw_full_matches"],
            match_trackers[label]["raw_match_dist_acc"],
            num_episodes,
        )
        suffix = _reference_suffix(label, policy)
        if label == "__main__":
            summary.update(tmp)
            if suffix:
                for key, value in tmp.items():
                    if key.endswith("_std"):
                        base = key[:-4]
                        summary[f"{base}_{suffix}_std"] = value
                    else:
                        summary[f"{key}_{suffix}"] = value
        elif suffix:
            for key, value in tmp.items():
                if key.endswith("_std"):
                    base = key[:-4]
                    summary[f"{base}_{suffix}_std"] = value
                else:
                    summary[f"{key}_{suffix}"] = value

    return summary
