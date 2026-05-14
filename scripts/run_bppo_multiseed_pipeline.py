import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_seeds(text: str):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def manifest_tag(item: dict):
    if item.get("dataset_tag"):
        return str(item["dataset_tag"])
    optimal_fraction = float(item["optimal_fraction"])
    return f"of_{optimal_fraction:.2f}".replace(".", "p")


def load_single_manifest_item(repo_root: Path, config: dict):
    manifest_path = repo_root / config["manifest_path"]
    manifest = load_json(manifest_path)
    if len(manifest) != 1:
        raise ValueError(
            f"Expected exactly one manifest entry in {manifest_path}, found {len(manifest)}"
        )
    return manifest[0]


def build_train_command(
    repo_root: Path,
    config_path: Path,
    seed_output_root: Path,
    seed: int,
):
    return [
        sys.executable,
        str(repo_root / "scripts" / "run_bppo_optfrac_sweep.py"),
        "--config",
        str(config_path),
        "--output_root",
        str(seed_output_root),
        "--seed",
        str(seed),
    ]


def build_select_command(repo_root: Path, run_dir: Path):
    return [
        sys.executable,
        str(repo_root / "scripts" / "select_bppo_checkpoint.py"),
        "--run_dir",
        str(run_dir),
        "--output_json",
        str(run_dir / "selected_checkpoint.json"),
    ]


def build_reeval_command(
    repo_root: Path,
    config: dict,
    run_dir: Path,
    num_machines: int,
    reeval_episodes: int,
):
    selection = load_json(run_dir / "selected_checkpoint.json")
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "evaluate_checkpoint.py"),
        "--checkpoint_dir",
        str(selection["selected_checkpoint_dir"]),
        "--output_json",
        str(run_dir / f"selected_checkpoint_eval_{reeval_episodes}.json"),
        "--env_name",
        str(config["env_name"]),
        "--num_machines",
        str(num_machines),
        "--horizon",
        str(config["horizon"]),
        "--episodes",
        str(reeval_episodes),
        "--discount",
        str(config["discount"]),
        "--device",
        str(config["device"]),
    ]
    if config.get("policy_path"):
        cmd.extend(["--policy_path", str(config["policy_path"])])
    if config.get("extra_policy_paths"):
        cmd.extend(["--extra_policy_paths", str(config["extra_policy_paths"])])
    return cmd


def aggregate_results(output_root: Path, seeds, num_machines: int, run_tag: str, reeval_episodes: int):
    per_seed = []
    for seed in seeds:
        run_dir = output_root / f"seed_{seed}" / f"MM{num_machines}" / run_tag
        selection = load_json(run_dir / "selected_checkpoint.json")
        reeval = load_json(run_dir / f"selected_checkpoint_eval_{reeval_episodes}.json")
        per_seed.append(
            {
                "seed": int(seed),
                "selected_step": int(selection["selected_step"]),
                "selected_discounted_cost_eval": float(selection["selected_discounted_cost"]),
                "selected_discounted_return_eval": float(selection["selected_discounted_return"]),
                "reeval_discounted_cost_mean": float(reeval["discounted_cost"]),
                "reeval_discounted_cost_std": float(reeval.get("discounted_cost_std", 0.0)),
                "reeval_discounted_return_mean": float(reeval["discounted_return"]),
                "reeval_discounted_return_std": float(reeval.get("discounted_return_std", 0.0)),
            }
        )

    cost_means = [row["reeval_discounted_cost_mean"] for row in per_seed]
    return_means = [row["reeval_discounted_return_mean"] for row in per_seed]
    summary = {
        "num_seeds": len(per_seed),
        "seeds": [int(seed) for seed in seeds],
        "metric_definition": (
            "For each seed, train with the unchanged BPPO configuration; use the "
            "training-produced best BPPO checkpoint selected by minimum discounted "
            f"evaluation cost; then re-evaluate that checkpoint with {reeval_episodes} episodes."
        ),
        "discounted_cost_mean_across_seeds": float(statistics.mean(cost_means)),
        "discounted_cost_std_across_seeds": float(statistics.stdev(cost_means)) if len(cost_means) > 1 else 0.0,
        "discounted_return_mean_across_seeds": float(statistics.mean(return_means)),
        "discounted_return_std_across_seeds": float(statistics.stdev(return_means)) if len(return_means) > 1 else 0.0,
        "per_seed": per_seed,
    }
    with open(output_root / "summary_multiseed.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run BPPO multi-seed training and large-rollout re-evaluation using the existing best-checkpoint rule."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="123,234,345")
    parser.add_argument("--reeval_episodes", type=int, default=1000)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_reeval", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    manifest_item = load_single_manifest_item(repo_root, config)
    run_tag = str(config.get("run_tag_override") or manifest_tag(manifest_item))
    num_machines = int(manifest_item["num_machines"])
    seeds = parse_seeds(args.seeds)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[JOB] config={config_path.name}")
    print(f"[JOB] output_root={output_root}")
    print(f"[JOB] seeds={seeds}")

    for seed in seeds:
        seed_output_root = output_root / f"seed_{seed}"
        run_dir = seed_output_root / f"MM{num_machines}" / run_tag
        print(f"[SEED {seed}] run_dir={run_dir}")

        if not args.skip_train:
            print(f"[SEED {seed}] training")
            train_cmd = build_train_command(repo_root, config_path, seed_output_root, seed)
            subprocess.run(train_cmd, cwd=repo_root, check=True)

        print(f"[SEED {seed}] selecting checkpoint")
        select_cmd = build_select_command(repo_root, run_dir)
        subprocess.run(select_cmd, cwd=repo_root, check=True)

        if not args.skip_reeval:
            print(f"[SEED {seed}] reevaluating selected checkpoint with {args.reeval_episodes} episodes")
            reeval_cmd = build_reeval_command(
                repo_root=repo_root,
                config=config,
                run_dir=run_dir,
                num_machines=num_machines,
                reeval_episodes=args.reeval_episodes,
            )
            subprocess.run(reeval_cmd, cwd=repo_root, check=True)

    if not args.skip_reeval:
        print("[JOB] aggregating results")
        aggregate_results(output_root, seeds, num_machines, run_tag, args.reeval_episodes)
        print(f"[JOB] summary written to {output_root / 'summary_multiseed.json'}")


if __name__ == "__main__":
    main()
