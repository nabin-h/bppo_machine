import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_thresholds(raw: str):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def mix_tag(thresholds, random_fraction: float):
    thr_tag = "_".join(f"t{threshold}" for threshold in thresholds)
    rnd_tag = f"rnd{int(round(random_fraction * 100))}"
    return f"mix_{thr_tag}_equal_{rnd_tag}"


def run(cmd, cwd: Path):
    print("[CMD]", " ".join(str(item) for item in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_threshold_policies(repo_root: Path, num_machines: int, env_name: str, thresholds, policies_dir: Path, force: bool):
    policy_paths = []
    for threshold in thresholds:
        policy_path = policies_dir / f"M{num_machines}_{env_name}_threshold{threshold}_policy.pkl"
        policy_paths.append(policy_path)
        if policy_path.exists() and not force:
            continue
        cmd = [sys.executable, str(repo_root / "scripts" / "create_component_threshold_policy.py"), "--output_path", str(policy_path), "--threshold", str(threshold), "--env_name", env_name, "--num_machines", str(num_machines)]
        run(cmd, cwd=repo_root)
    return policy_paths


def ensure_dataset(repo_root: Path, num_machines: int, env_name: str, thresholds, random_fraction: float, dataset_root: Path, num_episodes: int, sim_timesteps: int, store_timesteps: int, gamma: float, rank_method: int, dataset_seed: int, force: bool):
    tag = mix_tag(thresholds, random_fraction)
    manifest_path = dataset_root / f"dataset_manifest_MM{num_machines}_{tag}.json"
    if not manifest_path.exists() or force:
        cmd = [sys.executable, str(repo_root / "scripts" / "create_env3_mixed_dataset.py"), "--num_machines", str(num_machines), "--num_episodes", str(num_episodes), "--sim_timesteps", str(sim_timesteps), "--store_timesteps", str(store_timesteps), "--thresholds", ",".join(str(item) for item in thresholds), "--random_fraction", str(random_fraction), "--output_root", str(dataset_root), "--gamma", str(gamma), "--rank_method", str(rank_method), "--seed", str(dataset_seed), "--env_name", env_name]
        run(cmd, cwd=repo_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected manifest at {manifest_path}")
    return manifest_path


def build_config_payload(env_name: str, manifest_path: Path, dataset_root_rel: str, policy_paths, device: str):
    primary_index = 1 if len(policy_paths) > 1 else 0
    for idx, path in enumerate(policy_paths):
        if "threshold6" in path.name:
            primary_index = idx
            break
    return {
        "manifest_path": str(Path(dataset_root_rel) / manifest_path.name),
        "repo_root": ".",
        "output_root": "Seed runs/M40 runs/runs_mm40_env3_567_r20_bppo_seedruns",
        "run_tag_override": "mix_t567_e_r20",
        "dataset_root": dataset_root_rel,
        "policy_path": str(Path("policies") / policy_paths[primary_index].name),
        "extra_policy_paths": ",".join(str(Path("policies") / path.name) for idx, path in enumerate(policy_paths) if idx != primary_index),
        "env_name": env_name,
        "seed": 123,
        "horizon": 100,
        "episodes": 100,
        "discount": 0.95,
        "device": device,
        "v_steps": 20000,
        "v_hidden_dim": 256,
        "v_depth": 2,
        "v_lr": 0.0001,
        "v_batch_size": 512,
        "q_bc_steps": 20000,
        "q_hidden_dim": 256,
        "q_depth": 2,
        "q_lr": 0.0001,
        "q_batch_size": 512,
        "target_update_freq": 2,
        "tau": 0.005,
        "bc_steps": 10000,
        "bc_hidden_dim": 256,
        "bc_depth": 2,
        "bc_lr": 0.0001,
        "bc_batch_size": 512,
        "bppo_steps": 5000,
        "bppo_hidden_dim": 256,
        "bppo_depth": 2,
        "bppo_lr": 0.0001,
        "bppo_batch_size": 512,
        "clip_ratio": 0.25,
        "entropy_weight": 0.0,
        "decay": 0.96,
        "omega": 0.9,
        "eval_interval": 100,
        "checkpoint_interval": 500,
        "log_interval": 100
    }


def main():
    parser = argparse.ArgumentParser(description="Full BPPO M40 env3 pipeline: threshold-policy creation, mixed dataset creation, multiseed training, summary, and plots.")
    parser.add_argument("--num_machines", type=int, default=40)
    parser.add_argument("--env_name", type=str, default="env3")
    parser.add_argument("--thresholds", type=str, default="5,6,7")
    parser.add_argument("--random_fraction", type=float, default=0.20)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="123,234,345")
    parser.add_argument("--num_episodes", type=int, default=20000)
    parser.add_argument("--sim_timesteps", type=int, default=100)
    parser.add_argument("--store_timesteps", type=int, default=100)
    parser.add_argument("--reeval_episodes", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--rank_method", type=int, default=2)
    parser.add_argument("--base_output_dir", type=str, default="Seed runs")
    parser.add_argument("--dataset_root", type=str, default="datasets_mm40_env3_mix567_r20")
    parser.add_argument("--policies_dir", type=str, default="policies")
    parser.add_argument("--generated_config_dir", type=str, default="configs/generated")
    parser.add_argument("--force_policies", action="store_true")
    parser.add_argument("--force_dataset", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    thresholds = parse_thresholds(args.thresholds)
    policies_dir = repo_root / args.policies_dir
    dataset_root = repo_root / args.dataset_root
    generated_config_dir = repo_root / args.generated_config_dir
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    policy_paths = ensure_threshold_policies(repo_root, args.num_machines, args.env_name, thresholds, policies_dir, args.force_policies)
    manifest_path = ensure_dataset(repo_root, args.num_machines, args.env_name, thresholds, args.random_fraction, dataset_root, args.num_episodes, args.sim_timesteps, args.store_timesteps, args.gamma, args.rank_method, args.dataset_seed, args.force_dataset)

    config_path = generated_config_dir / "mm40_env3_567_r20_seedruns.json"
    config_payload = build_config_payload(args.env_name, manifest_path, args.dataset_root, policy_paths, args.device)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    print(f"[INFO] wrote config {config_path}")

    output_root = repo_root / args.base_output_dir / "M40 runs" / "runs_mm40_env3_567_r20_bppo_seedruns"
    cmd = [sys.executable, str(repo_root / "scripts" / "run_bppo_multiseed_pipeline.py"), "--config", str(config_path), "--output_root", str(output_root), "--seeds", args.seeds, "--reeval_episodes", str(args.reeval_episodes)]
    run(cmd, cwd=repo_root)

    plot_cmd = [sys.executable, str(repo_root / "scripts" / "plot_bppo_multiseed_cost_curves.py"), "--output_root", str(output_root), "--run_tag", "mix_t567_e_r20"]
    run(plot_cmd, cwd=repo_root)

    print("[DONE]")
    print(f"[DONE] output_root={output_root}")
    print(f"[DONE] manifest_path={manifest_path}")


if __name__ == "__main__":
    main()
