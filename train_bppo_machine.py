import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from binary_bppo import BehaviorCloning, BehaviorProximalPolicyOptimization
from buffer import OfflineReplayBuffer
from critic import QSarsaLearner, ValueLearner
from custom_env import (DEFAULT_ENV_NAME, MachineMaintenanceEnv, get_env_spec,
                        random_viable_action, threshold_policy)
from evaluation import (evaluate_policy_fn, load_reference_policy,
                        reference_action_for_state)
from maintenance_dataset import load_trajectory_dataset


def parse_int_list(text: str):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def load_extra_reference_policies(raw_paths):
    extra_reference_policies = {}
    expanded_paths = []
    for item in raw_paths or []:
        if not item:
            continue
        expanded_paths.extend([part.strip() for part in str(item).split(",") if part.strip()])
    for policy_path in expanded_paths:
        if not policy_path:
            continue
        policy = load_reference_policy(policy_path)
        if isinstance(policy, dict) and policy.get("policy_type") == "component_threshold":
            label = f"t{int(policy.get('threshold', 2))}"
        else:
            label = Path(policy_path).stem
        extra_reference_policies[label] = policy
    return extra_reference_policies


def make_env(num_machines, horizon, seed, env_name):
    env = MachineMaintenanceEnv(
        num_machines=num_machines,
        horizon=horizon,
        reward_is_negative_cost=True,
        seed=seed,
        env_name=env_name,
    )
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def summarize_eval(step, eval_stats, checkpoint_dir=None, is_best=False, stage=None, cumulative_step=None):
    row = {
        "stage": stage,
        "step": int(step),
        "cumulative_step": int(cumulative_step) if cumulative_step is not None else None,
        "return": float(eval_stats["discounted_return"]),
        "return_std": float(eval_stats.get("discounted_return_std", np.nan)),
        "undiscounted_return": float(eval_stats["return"]),
        "undiscounted_return_std": float(eval_stats.get("return_std", np.nan)),
        "length": float(eval_stats["length"]),
        "length_std": float(eval_stats.get("length_std", np.nan)),
        "cost": float(eval_stats.get("discounted_cost", np.nan)),
        "cost_std": float(eval_stats.get("discounted_cost_std", np.nan)),
        "undiscounted_cost": float(eval_stats.get("cost", np.nan)),
        "undiscounted_cost_std": float(eval_stats.get("cost_std", np.nan)),
        "action_ones_per_step": float(eval_stats.get("action_ones_per_step", np.nan)),
        "action_ones_per_step_std": float(eval_stats.get("action_ones_per_step_std", np.nan)),
        "action_activation_rate": float(eval_stats.get("action_activation_rate", np.nan)),
        "action_activation_rate_std": float(eval_stats.get("action_activation_rate_std", np.nan)),
        "corrections": float(eval_stats.get("corrections", np.nan)),
        "corrections_std": float(eval_stats.get("corrections_std", np.nan)),
        "raw_invalid_action_count": float(eval_stats.get("raw_invalid_action_count", np.nan)),
        "raw_invalid_action_count_std": float(eval_stats.get("raw_invalid_action_count_std", np.nan)),
        "raw_invalid_action_rate": float(eval_stats.get("raw_invalid_action_rate", np.nan)),
        "raw_invalid_action_rate_std": float(eval_stats.get("raw_invalid_action_rate_std", np.nan)),
        "full_match_pct": float(eval_stats.get("full_match_pct", np.nan)),
        "full_match_pct_std": float(eval_stats.get("full_match_pct_std", np.nan)),
        "projected_full_match_pct": float(eval_stats.get("projected_full_match_pct", np.nan)),
        "projected_full_match_pct_std": float(eval_stats.get("projected_full_match_pct_std", np.nan)),
        "raw_full_match_pct": float(eval_stats.get("raw_full_match_pct", np.nan)),
        "raw_full_match_pct_std": float(eval_stats.get("raw_full_match_pct_std", np.nan)),
        "match_distribution": eval_stats.get("match_distribution"),
        "projected_match_distribution": eval_stats.get("projected_match_distribution"),
        "raw_match_distribution": eval_stats.get("raw_match_distribution"),
        "checkpoint_dir": checkpoint_dir,
        "is_best_so_far": bool(is_best),
    }
    for key, value in eval_stats.items():
        if key in row:
            continue
        if key.endswith("_std") or key.startswith("full_match_pct_") or key.startswith("projected_full_match_pct_") or key.startswith("raw_full_match_pct_") or key.startswith("match_distribution_") or key.startswith("projected_match_distribution_") or key.startswith("raw_match_distribution_"):
            row[key] = value
    return row


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_policy_checkpoint(base_dir: Path, step: int, policy, mean, std, metadata):
    ckpt_dir = base_dir / "checkpoints" / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save(str(ckpt_dir / "policy.pt"))
    np.savez(ckpt_dir / "normalization.npz", mean=mean, std=std)
    save_json(ckpt_dir / "metadata.json", metadata)
    return ckpt_dir


def make_policy_fn(policy, mean, std, device):
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)

    def policy_fn(observation):
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        obs = (obs - mean_arr) / std_arr
        tensor_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = policy.select_action(tensor_obs, is_sample=False)
        return action.cpu().numpy().reshape(-1).astype(np.int32)

    return policy_fn


def main():
    parser = argparse.ArgumentParser(description="Train BPPO on the maintenance benchmark.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--env_name", type=str, default=DEFAULT_ENV_NAME)
    parser.add_argument("--num_machines", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--is_state_norm", action="store_true")
    parser.add_argument("--policy_path", type=str, default=None)
    parser.add_argument("--extra_policy_paths", type=str, nargs="*", default=None)

    parser.add_argument("--v_steps", type=int, default=20000)
    parser.add_argument("--v_hidden_dim", type=int, default=256)
    parser.add_argument("--v_depth", type=int, default=2)
    parser.add_argument("--v_lr", type=float, default=1e-4)
    parser.add_argument("--v_batch_size", type=int, default=512)

    parser.add_argument("--q_bc_steps", type=int, default=20000)
    parser.add_argument("--q_hidden_dim", type=int, default=256)
    parser.add_argument("--q_depth", type=int, default=2)
    parser.add_argument("--q_lr", type=float, default=1e-4)
    parser.add_argument("--q_batch_size", type=int, default=512)
    parser.add_argument("--target_update_freq", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--bc_steps", type=int, default=10000)
    parser.add_argument("--bc_hidden_dim", type=int, default=256)
    parser.add_argument("--bc_depth", type=int, default=2)
    parser.add_argument("--bc_lr", type=float, default=1e-4)
    parser.add_argument("--bc_batch_size", type=int, default=512)

    parser.add_argument("--bppo_steps", type=int, default=5000)
    parser.add_argument("--bppo_hidden_dim", type=int, default=256)
    parser.add_argument("--bppo_depth", type=int, default=2)
    parser.add_argument("--bppo_lr", type=float, default=1e-4)
    parser.add_argument("--bppo_batch_size", type=int, default=512)
    parser.add_argument("--clip_ratio", type=float, default=0.25)
    parser.add_argument("--entropy_weight", type=float, default=0.0)
    parser.add_argument("--decay", type=float, default=0.96)
    parser.add_argument("--omega", type=float, default=0.9)
    parser.add_argument("--is_clip_decay", action="store_true")
    parser.add_argument("--is_bppo_lr_decay", action="store_true")
    parser.add_argument("--is_update_old_policy", action="store_true")

    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.num_machines, args.horizon, args.seed, args.env_name)
    action_dim = int(np.asarray(env.action_space.sample()).shape[-1])
    state_dim = int(np.asarray(env.observation_space.sample()).shape[-1])

    dataset = load_trajectory_dataset(args.dataset_path, reward_is_negative_cost=True)
    replay_buffer = OfflineReplayBuffer(device, state_dim, action_dim, len(dataset["actions"]))
    replay_buffer.load_maintenance_dataset(dataset)
    replay_buffer.compute_return(args.discount)
    if args.is_state_norm:
        mean, std = replay_buffer.normalize_state()
    else:
        mean = np.zeros((1, state_dim), dtype=np.float32)
        std = np.ones((1, state_dim), dtype=np.float32)

    reference_policy = load_reference_policy(args.policy_path) if args.policy_path else None
    extra_reference_policies = load_extra_reference_policies(args.extra_policy_paths)

    threshold_value = int(
        reference_policy.get("threshold", get_env_spec(args.env_name)["threshold"])
    ) if reference_policy else int(get_env_spec(args.env_name)["threshold"])

    baseline_random = evaluate_policy_fn(
        lambda obs: random_viable_action(obs, env_name=args.env_name),
        env,
        args.episodes,
        discount=args.discount,
        reference_policy=reference_policy,
        extra_reference_policies=extra_reference_policies,
        env_name=args.env_name,
    )
    baseline_threshold = evaluate_policy_fn(
        lambda obs: threshold_policy(obs, threshold=threshold_value, env_name=args.env_name),
        env,
        args.episodes,
        discount=args.discount,
        reference_policy=reference_policy,
        extra_reference_policies=extra_reference_policies,
        env_name=args.env_name,
    )
    baseline_payload = {
        "random_viable": baseline_random,
        "threshold": baseline_threshold,
        "env_name": args.env_name,
        "threshold_value": threshold_value,
    }
    save_json(save_dir / "baseline_eval.json", baseline_payload)

    value = ValueLearner(device, state_dim, args.v_hidden_dim, args.v_depth, args.v_lr, args.v_batch_size)
    q_bc = QSarsaLearner(
        device, state_dim, action_dim, args.q_hidden_dim, args.q_depth,
        args.q_lr, args.target_update_freq, args.tau, args.discount, args.q_batch_size,
    )
    bc = BehaviorCloning(
        device, state_dim, args.bc_hidden_dim, args.bc_depth,
        action_dim, args.bc_lr, args.bc_batch_size,
    )
    bppo = BehaviorProximalPolicyOptimization(
        device, state_dim, args.bppo_hidden_dim, args.bppo_depth, action_dim,
        args.bppo_lr, args.clip_ratio, args.entropy_weight, args.decay,
        args.omega, args.bppo_batch_size,
    )

    loss_history = []
    eval_history = []
    best_payload = None
    best_cost = np.inf
    start_time = time.perf_counter()
    value_offset = 0
    q_bc_offset = args.v_steps
    bc_offset = args.v_steps + args.q_bc_steps
    bppo_offset = args.v_steps + args.q_bc_steps + args.bc_steps

    for step in range(1, args.v_steps + 1):
        value_loss = value.update(replay_buffer)
        if step % args.log_interval == 0 or step == args.v_steps:
            loss_history.append({
                "stage": "value",
                "step": step,
                "cumulative_step": value_offset + step,
                "value_loss": float(value_loss),
            })

    for step in range(1, args.q_bc_steps + 1):
        q_loss = q_bc.update(replay_buffer, pi=None)
        if step % args.log_interval == 0 or step == args.q_bc_steps:
            loss_history.append({
                "stage": "q_bc",
                "step": step,
                "cumulative_step": q_bc_offset + step,
                "q_loss": float(q_loss),
            })

    best_bc_path = save_dir / "bc_best.pt"
    best_bc_cost = np.inf
    for step in range(1, args.bc_steps + 1):
        bc_loss = bc.update(replay_buffer)
        if step % args.log_interval == 0 or step == args.bc_steps:
            loss_history.append({
                "stage": "bc",
                "step": step,
                "cumulative_step": bc_offset + step,
                "bc_loss": float(bc_loss),
            })
        if step % args.eval_interval == 0 or step == args.bc_steps:
            bc_policy_fn = make_policy_fn(bc, mean, std, device)
            eval_stats = evaluate_policy_fn(
                bc_policy_fn,
                env,
                args.episodes,
                discount=args.discount,
                reference_policy=reference_policy,
                extra_reference_policies=extra_reference_policies,
                env_name=args.env_name,
            )
            cost = float(eval_stats["discounted_cost"])
            eval_history.append(
                summarize_eval(
                    step,
                    eval_stats,
                    checkpoint_dir=None,
                    is_best=cost < best_bc_cost,
                    stage="bc",
                    cumulative_step=bc_offset + step,
                )
            )
            if cost < best_bc_cost:
                best_bc_cost = cost
                bc.save(str(best_bc_path))

    bppo.load(str(best_bc_path))
    bppo.set_old_policy()

    for step in range(1, args.bppo_steps + 1):
        bppo_loss = bppo.update(
            replay_buffer,
            q_bc,
            value,
            args.is_clip_decay and step <= 200,
            args.is_bppo_lr_decay and step <= 200,
        )
        if step % args.log_interval == 0 or step == args.bppo_steps:
            loss_history.append({
                "stage": "bppo",
                "step": step,
                "cumulative_step": bppo_offset + step,
                "bppo_loss": float(bppo_loss),
            })

        if step % args.eval_interval == 0 or step == args.bppo_steps:
            policy_fn = make_policy_fn(bppo, mean, std, device)
            eval_stats = evaluate_policy_fn(
                policy_fn,
                env,
                args.episodes,
                discount=args.discount,
                reference_policy=reference_policy,
                extra_reference_policies=extra_reference_policies,
                env_name=args.env_name,
            )
            checkpoint_dir = str(save_policy_checkpoint(
                save_dir,
                step,
                bppo,
                mean,
                std,
                {
                    "env_name": args.env_name,
                    "num_machines": args.num_machines,
                    "horizon": args.horizon,
                    "seed": args.seed,
                    "checkpoint_step": step,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "policy_hidden_dim": args.bppo_hidden_dim,
                    "policy_depth": args.bppo_depth,
                    "state_norm": bool(args.is_state_norm),
                },
            ))
            is_best = False
            row = summarize_eval(step, eval_stats, checkpoint_dir=checkpoint_dir, is_best=False)
            row["stage"] = "bppo"
            row["cumulative_step"] = bppo_offset + step
            eval_history.append(row)
            if float(eval_stats["discounted_cost"]) < best_cost:
                best_cost = float(eval_stats["discounted_cost"])
                is_best = True
                row["is_best_so_far"] = True
                best_payload = {
                    "best_step": int(step),
                    "best_return": float(-best_cost),
                    "best_cost": float(best_cost),
                    "checkpoint_dir": checkpoint_dir,
                }
                save_json(save_dir / "best_checkpoint.json", best_payload)
                if checkpoint_dir:
                    latest_best_dir = save_dir / "checkpoints" / "best"
                    latest_best_dir.mkdir(parents=True, exist_ok=True)
                    bppo.save(str(latest_best_dir / "policy.pt"))
                    np.savez(latest_best_dir / "normalization.npz", mean=mean, std=std)
                    save_json(latest_best_dir / "metadata.json", {
                        "best_step": int(step),
                        "source_checkpoint_dir": checkpoint_dir,
                    })
                if args.is_update_old_policy:
                    bppo.set_old_policy()

    total_wall_time = time.perf_counter() - start_time
    save_json(save_dir / f"{args.seed}_loss.json", loss_history)
    save_json(save_dir / f"{args.seed}_eval.json", eval_history)
    save_json(save_dir / "timing.json", {
        "seed": args.seed,
        "train_wall_time_sec": total_wall_time,
        "env_name": args.env_name,
        "dataset_path": args.dataset_path,
        "v_steps": args.v_steps,
        "q_bc_steps": args.q_bc_steps,
        "bc_steps": args.bc_steps,
        "bppo_steps": args.bppo_steps,
    })
    if best_payload is None:
        save_json(save_dir / "best_checkpoint.json", {
            "best_step": None,
            "best_return": None,
            "best_cost": None,
            "checkpoint_dir": None,
        })


if __name__ == "__main__":
    main()
