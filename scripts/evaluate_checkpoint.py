import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from binary_bppo import BehaviorProximalPolicyOptimization
from custom_env import DEFAULT_ENV_NAME, MachineMaintenanceEnv
from evaluation import evaluate_policy_fn, load_reference_policy


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


def build_policy(checkpoint_dir: Path, device: torch.device):
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    norm = np.load(checkpoint_dir / "normalization.npz")
    mean = norm["mean"].astype(np.float32)
    std = norm["std"].astype(np.float32)

    policy = BehaviorProximalPolicyOptimization(
        device=device,
        state_dim=int(metadata["state_dim"]),
        hidden_dim=int(metadata["policy_hidden_dim"]),
        depth=int(metadata["policy_depth"]),
        action_dim=int(metadata["action_dim"]),
        policy_lr=1e-4,
        clip_ratio=0.25,
        entropy_weight=0.0,
        decay=0.96,
        omega=0.9,
        batch_size=512,
    )
    policy.load(str(checkpoint_dir / "policy.pt"))
    return policy, mean, std, metadata


def make_policy_fn(policy, mean, std, device):
    def policy_fn(observation):
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        obs = (obs - mean) / std
        tensor_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = policy.select_action(tensor_obs, is_sample=False)
        return action.cpu().numpy().reshape(-1).astype(np.int32)

    return policy_fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved BPPO checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--env_name", type=str, default=DEFAULT_ENV_NAME)
    parser.add_argument("--num_machines", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--policy_path", type=str, default=None)
    parser.add_argument("--extra_policy_paths", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_dir = Path(args.checkpoint_dir)
    policy, mean, std, _ = build_policy(checkpoint_dir, device)
    policy_fn = make_policy_fn(policy, mean, std, device)

    env = MachineMaintenanceEnv(
        num_machines=args.num_machines,
        horizon=args.horizon,
        reward_is_negative_cost=True,
        seed=0,
        env_name=args.env_name,
    )

    reference_policy = load_reference_policy(args.policy_path) if args.policy_path else None
    extra_reference_policies = load_extra_reference_policies(args.extra_policy_paths)
    result = evaluate_policy_fn(
        policy_fn,
        env,
        args.episodes,
        discount=args.discount,
        reference_policy=reference_policy,
        extra_reference_policies=extra_reference_policies,
        env_name=args.env_name,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
