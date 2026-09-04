from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ENV_NAME = "gamma_rul_v1"
THRESHOLDS = (6, 7, 8, 9)
RANDOM_FRACTION = 0.20


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def run(command: list[str], cwd: Path) -> None:
    print("[CMD]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def expected_dataset_paths(
    dataset_root: Path,
    num_machines: int,
    num_episodes: int,
) -> tuple[Path, Path, str]:
    threshold_tag = "_".join(f"t{value}" for value in THRESHOLDS)
    run_tag = f"mix_{threshold_tag}_equal_rnd20"
    dataset_dir = dataset_root / f"M{num_machines}" / run_tag
    dataset_path = dataset_dir / (
        f"traj_gamma_rul_M{num_machines}_{run_tag}_{num_episodes}.pkl"
    )
    return dataset_path, dataset_dir / "dataset_metadata.json", run_tag


def validate_dataset(
    dataset_path: Path,
    metadata_path: Path,
    num_machines: int,
    horizon: int,
    discount: float,
) -> dict:
    if not dataset_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Copy the DT gamma dataset and dataset_metadata.json into the BPPO "
            f"repository first. Expected: {dataset_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "num_machines": num_machines,
        "horizon": horizon,
        "thresholds": list(THRESHOLDS),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Dataset metadata mismatch for {key}: "
                f"expected {value}, found {metadata.get(key)}"
            )
    if abs(float(metadata.get("discount", -1.0)) - discount) > 1e-12:
        raise ValueError("Dataset discount does not match the requested discount.")
    if abs(float(metadata.get("random_fraction", -1.0)) - RANDOM_FRACTION) > 1e-12:
        raise ValueError("Dataset random fraction must be 0.20.")
    if metadata.get("reward_definition") != "reward = -cost":
        raise ValueError("BPPO requires the dataset rewards to equal negative cost.")
    environment = metadata.get("environment", {})
    if environment.get("env_name") != ENV_NAME:
        raise ValueError("Dataset environment is not gamma_rul_v1.")
    return metadata


def ensure_reference_policies(
    repo_root: Path,
    num_machines: int,
) -> tuple[Path, Path]:
    policies_dir = repo_root / "policies"
    threshold_path = policies_dir / f"M{num_machines}_{ENV_NAME}_threshold8.pkl"
    opportunity_path = policies_dir / f"M{num_machines}_{ENV_NAME}_t8_o6.pkl"
    if not threshold_path.exists():
        run(
            [
                sys.executable,
                str(repo_root / "scripts" / "create_component_threshold_policy.py"),
                "--output_path", str(threshold_path),
                "--threshold", "8",
                "--env_name", ENV_NAME,
                "--num_machines", str(num_machines),
            ],
            repo_root,
        )
    if not opportunity_path.exists():
        run(
            [
                sys.executable,
                str(repo_root / "scripts" / "create_opportunity_policy.py"),
                "--output_path", str(opportunity_path),
                "--trigger", "8",
                "--opportunity_threshold", "6",
                "--env_name", ENV_NAME,
                "--num_machines", str(num_machines),
            ],
            repo_root,
        )
    return threshold_path, opportunity_path


def write_config(
    repo_root: Path,
    dataset_path: Path,
    dataset_root: Path,
    threshold_path: Path,
    opportunity_path: Path,
    num_machines: int,
    run_tag: str,
    args: argparse.Namespace,
) -> Path:
    generated_dir = repo_root / "configs" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = generated_dir / f"M{num_machines}_{ENV_NAME}_manifest.json"
    manifest = [
        {
            "dataset_tag": run_tag,
            "dataset_path": str(dataset_path.resolve()),
            "num_machines": num_machines,
        }
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    config = {
        "manifest_path": str(manifest_path.resolve()),
        "repo_root": str(repo_root.resolve()),
        "output_root": str(
            repo_root / "Seed runs" / ENV_NAME / f"M{num_machines}"
            / args.run_label
        ),
        "run_tag_override": run_tag,
        "dataset_root": str(dataset_root.resolve()),
        "policy_path": str(threshold_path.resolve()),
        "extra_policy_paths": str(opportunity_path.resolve()),
        "env_name": ENV_NAME,
        "horizon": args.horizon,
        "episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "discount": args.discount,
        "reward_scale": args.reward_scale,
        "device": args.device,
        "reeval_seed": args.reeval_seed,
        "is_state_norm": args.is_state_norm,
        "v_steps": args.v_steps,
        "v_hidden_dim": args.hidden_dim,
        "v_depth": args.depth,
        "v_lr": args.learning_rate,
        "v_batch_size": args.batch_size,
        "q_bc_steps": args.q_steps,
        "q_hidden_dim": args.hidden_dim,
        "q_depth": args.depth,
        "q_lr": args.learning_rate,
        "q_batch_size": args.batch_size,
        "target_update_freq": 2,
        "tau": 0.005,
        "bc_steps": args.bc_steps,
        "bc_checkpoint": args.bc_checkpoint,
        "bc_hidden_dim": args.hidden_dim,
        "bc_depth": args.depth,
        "bc_lr": args.learning_rate,
        "bc_batch_size": args.batch_size,
        "bppo_steps": args.bppo_steps,
        "bppo_hidden_dim": args.hidden_dim,
        "bppo_depth": args.depth,
        "bppo_lr": args.learning_rate,
        "bppo_batch_size": args.batch_size,
        "clip_ratio": args.clip_ratio,
        "entropy_weight": args.entropy_weight,
        "decay": 0.96,
        "omega": args.omega,
        "eval_interval": args.eval_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "log_interval": args.log_interval,
    }
    config_path = generated_dir / (
        f"M{num_machines}_{ENV_NAME}_{args.run_label}.json"
    )
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BPPO on existing gamma_rul_v1 datasets."
    )
    parser.add_argument("--systems", default="10")
    parser.add_argument("--seeds", default="123,234,345")
    parser.add_argument("--dataset_root", default="datasets_gamma_rul_v1")
    parser.add_argument("--num_episodes", type=int, default=20000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--eval_seed", type=int, default=10000)
    parser.add_argument("--reeval_episodes", type=int, default=1000)
    parser.add_argument("--reeval_seed", type=int, default=0)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--run_label", default="bppo_multiseed")
    parser.add_argument("--v_steps", type=int, default=20000)
    parser.add_argument("--q_steps", type=int, default=20000)
    parser.add_argument("--bc_steps", type=int, default=10000)
    parser.add_argument("--bc_checkpoint", default=None)
    parser.add_argument("--bppo_steps", type=int, default=5000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--clip_ratio", type=float, default=0.25)
    parser.add_argument("--entropy_weight", type=float, default=0.0)
    parser.add_argument("--omega", type=float, default=0.9)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--is_state_norm", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_reeval", action="store_true")
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dataset_root = (repo_root / args.dataset_root).resolve()
    for num_machines in parse_ints(args.systems):
        dataset_path, metadata_path, run_tag = expected_dataset_paths(
            dataset_root, num_machines, args.num_episodes
        )
        validate_dataset(
            dataset_path, metadata_path, num_machines, args.horizon, args.discount
        )
        threshold_path, opportunity_path = ensure_reference_policies(
            repo_root, num_machines
        )
        config_path = write_config(
            repo_root,
            dataset_path,
            dataset_root,
            threshold_path,
            opportunity_path,
            num_machines,
            run_tag,
            args,
        )
        output_root = (
            repo_root / "Seed runs" / ENV_NAME / f"M{num_machines}"
            / args.run_label
        )
        command = [
            sys.executable,
            str(repo_root / "scripts" / "run_bppo_multiseed_pipeline.py"),
            "--config", str(config_path),
            "--output_root", str(output_root),
            "--seeds", args.seeds,
            "--reeval_episodes", str(args.reeval_episodes),
        ]
        if args.skip_train:
            command.append("--skip_train")
        if args.skip_reeval:
            command.append("--skip_reeval")
        run(command, repo_root)

        if not args.skip_plot and not args.skip_reeval:
            run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "plot_bppo_multiseed_cost_curves.py"),
                    "--output_root", str(output_root),
                ],
                repo_root,
            )


if __name__ == "__main__":
    main()
