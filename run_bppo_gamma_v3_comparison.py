from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.gamma_v3_experiment import (
    dataset_files,
    environment_name,
    parse_csv,
    prepare_study,
    write_json,
)


def run(command: list[str], repo_root: Path) -> None:
    print("[CMD]", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def load_completed_summary(output_root: Path, seeds: list[int]) -> dict | None:
    summary_path = output_root / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    completed_seeds = sorted(int(row["seed"]) for row in summary.get("per_seed", []))
    return summary if completed_seeds == sorted(seeds) else None


def write_missing_log(
    path: Path,
    dt_repo: Path,
    missing_studies: list[dict],
) -> None:
    write_json(path, {
        "method": "BPPO",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "dt_repo": str(dt_repo),
        "missing_count": len(missing_studies),
        "missing_studies": missing_studies,
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BPPO on the exact gamma-v3 datasets used by DT."
    )
    parser.add_argument("--dt_repo", default="../dt_machine")
    parser.add_argument("--machine_types", default="homogeneous,heterogeneous")
    parser.add_argument("--setup_cost_per_machine", default="5,25")
    parser.add_argument("--num_machines", default="20,40")
    parser.add_argument("--modes", default="plausible,threshold")
    parser.add_argument("--num_episodes", type=int, default=20000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--seeds", default="123")
    parser.add_argument("--eval_episodes", type=int, default=25)
    parser.add_argument("--eval_seed", type=int, default=10000)
    parser.add_argument("--reeval_episodes", type=int, default=200)
    parser.add_argument("--reeval_seed", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--v_steps", type=int, default=20000)
    parser.add_argument("--q_bc_steps", type=int, default=20000)
    parser.add_argument("--bc_steps", type=int, default=10000)
    parser.add_argument("--bppo_steps", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_missing_datasets", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    dt_repo = (repo_root / args.dt_repo).resolve()
    machine_types = parse_csv(args.machine_types)
    setup_costs = parse_csv(args.setup_cost_per_machine, int)
    machine_counts = parse_csv(args.num_machines, int)
    modes = parse_csv(args.modes)
    seeds = parse_csv(args.seeds, int)
    suite_rows = []
    comparison_root = repo_root / "Seed runs" / "gamma_v3_comparison"
    missing_log_path = comparison_root / "bppo_missing_datasets.json"
    missing_studies = []
    write_missing_log(missing_log_path, dt_repo, missing_studies)

    for machine_type in machine_types:
        for setup_cost in setup_costs:
            for num_machines in machine_counts:
                for mode in modes:
                    env_name = environment_name(machine_type, setup_cost, num_machines)
                    output_root = (
                        comparison_root / env_name / f"M{num_machines}" / mode / "bppo"
                    )
                    if args.resume:
                        completed = load_completed_summary(output_root, seeds)
                        if completed is not None:
                            print(f"[RESUME] already complete: {output_root}", flush=True)
                            suite_rows.append(completed)
                            continue

                    try:
                        study = prepare_study(
                            repo_root,
                            dt_repo,
                            machine_type,
                            setup_cost,
                            num_machines,
                            mode,
                            args.num_episodes,
                            args.horizon,
                            args.discount,
                            args.dataset_seed,
                        )
                    except FileNotFoundError as error:
                        if not args.skip_missing_datasets:
                            raise
                        dataset_path, metadata_path = dataset_files(
                            dt_repo, env_name, num_machines, mode, args.num_episodes
                        )
                        missing_paths = [
                            str(path.resolve())
                            for path in (dataset_path, metadata_path)
                            if not path.exists()
                        ]
                        missing_studies.append({
                            "machine_type": machine_type,
                            "setup_cost_per_machine": setup_cost,
                            "num_machines": num_machines,
                            "mode": mode,
                            "environment": env_name,
                            "dataset_path": str(dataset_path.resolve()),
                            "metadata_path": str(metadata_path.resolve()),
                            "missing_paths": missing_paths,
                            "reason": str(error),
                        })
                        write_missing_log(missing_log_path, dt_repo, missing_studies)
                        print(f"[SKIP MISSING] {env_name} M{num_machines} {mode}", flush=True)
                        continue
                    prep = {
                        "method": "BPPO",
                        "environment": study.env_name,
                        "machine_type": machine_type,
                        "setup_cost_per_machine": setup_cost,
                        "num_machines": num_machines,
                        "mode": mode,
                        "dataset_path": str(study.dataset_path),
                        "metadata_path": str(study.metadata_path),
                        "training_seeds": seeds,
                        "checkpoint_eval_seed": args.eval_seed,
                        "checkpoint_eval_episodes": args.eval_episodes,
                    }
                    write_json(output_root / "experiment_spec.json", prep)
                    if args.prepare_only:
                        print(f"[PREPARED] {output_root}", flush=True)
                        continue

                    seed_rows = []
                    for seed in seeds:
                        run_dir = output_root / f"seed_{seed}"
                        if not args.skip_train:
                            run([
                                sys.executable,
                                str(repo_root / "train_gamma_v3_bppo.py"),
                                "--dataset_path", str(study.dataset_path),
                                "--save_dir", str(run_dir),
                                "--seed", str(seed),
                                "--env_name", study.env_name,
                                "--num_machines", str(num_machines),
                                "--horizon", str(args.horizon),
                                "--episodes", str(args.eval_episodes),
                                "--eval_seed", str(args.eval_seed),
                                "--discount", str(args.discount),
                                "--reward_scale", str(args.reward_scale),
                                "--device", args.device,
                                "--policy_path", str(study.policy_path),
                                "--extra_policy_paths",
                                *[str(path) for path in study.extra_policy_paths],
                                "--v_steps", str(args.v_steps),
                                "--v_hidden_dim", "256",
                                "--v_depth", "2",
                                "--v_lr", "0.0001",
                                "--v_batch_size", "512",
                                "--q_bc_steps", str(args.q_bc_steps),
                                "--q_hidden_dim", "256",
                                "--q_depth", "2",
                                "--q_lr", "0.0001",
                                "--q_batch_size", "512",
                                "--target_update_freq", "2",
                                "--tau", "0.005",
                                "--bc_steps", str(args.bc_steps),
                                "--bc_hidden_dim", "256",
                                "--bc_depth", "2",
                                "--bc_lr", "0.0001",
                                "--bc_batch_size", "512",
                                "--bppo_steps", str(args.bppo_steps),
                                "--bppo_hidden_dim", "256",
                                "--bppo_depth", "2",
                                "--bppo_lr", "0.0001",
                                "--bppo_batch_size", "512",
                                "--clip_ratio", "0.25",
                                "--entropy_weight", "0.0",
                                "--decay", "0.96",
                                "--omega", "0.9",
                                "--eval_interval", str(args.eval_interval),
                                "--checkpoint_interval", str(args.checkpoint_interval),
                                "--log_interval", "100",
                            ], repo_root)

                        selection_json = run_dir / "selected_checkpoint.json"
                        run([
                            sys.executable,
                            str(repo_root / "scripts" / "select_bppo_checkpoint.py"),
                            "--run_dir", str(run_dir),
                            "--output_json", str(selection_json),
                        ], repo_root)
                        selection = json.loads(selection_json.read_text(encoding="utf-8"))
                        reeval_json = run_dir / f"reeval_{args.reeval_episodes}.json"
                        run([
                            sys.executable,
                            str(repo_root / "evaluate_gamma_v3_bppo.py"),
                            "--checkpoint_dir", selection["selected_checkpoint_dir"],
                            "--output_json", str(reeval_json),
                            "--env_name", study.env_name,
                            "--num_machines", str(num_machines),
                            "--horizon", str(args.horizon),
                            "--episodes", str(args.reeval_episodes),
                            "--eval_seed", str(args.reeval_seed),
                            "--discount", str(args.discount),
                            "--policy_path", str(study.policy_path),
                            "--extra_policy_paths",
                            *[str(path) for path in study.extra_policy_paths],
                            "--device", args.device,
                        ], repo_root)
                        result = json.loads(reeval_json.read_text(encoding="utf-8"))
                        seed_rows.append({
                            "seed": seed,
                            "selected_step": int(selection["selected_step"]),
                            "discounted_cost": float(result["discounted_cost"]),
                            "episode_cost_std": float(result["discounted_cost_std"]),
                            "corrections": float(result.get("corrections", 0.0)),
                            "raw_invalid_action_rate": float(
                                result.get("raw_invalid_action_rate", 0.0)
                            ),
                        })

                    costs = [row["discounted_cost"] for row in seed_rows]
                    summary = {
                        **prep,
                        "reevaluation_episodes": args.reeval_episodes,
                        "reevaluation_seed": args.reeval_seed,
                        "checkpoint_selection": "minimum BPPO-stage evaluation cost",
                        "per_seed": seed_rows,
                        "discounted_cost_across_seeds": summarize(costs),
                    }
                    write_json(output_root / "summary.json", summary)
                    suite_rows.append(summary)

    if suite_rows:
        write_json(
            comparison_root / "bppo_suite_summary.json",
            {"method": "BPPO", "studies": suite_rows},
        )
    write_missing_log(missing_log_path, dt_repo, missing_studies)


if __name__ == "__main__":
    main()
