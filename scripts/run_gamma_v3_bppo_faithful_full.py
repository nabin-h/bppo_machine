from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.gamma_v3_experiment import (
    dataset_files,
    environment_name,
    parse_csv,
    prepare_study,
    write_json,
)
from scripts.run_gamma_v3_bppo_faithful_tuning import (
    ORIGINAL_BUDGETS,
    ORIGINAL_NETWORKS,
    evaluation_command,
    run,
    selected_metrics,
    training_command,
)


FROZEN_CLIP_RATIO = 0.25
FROZEN_DECAY = 0.96


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def completed_stage_checkpoint(run_dir: Path, stage: str, filename: str) -> Path | None:
    status_path = run_dir / "stage_status.json"
    checkpoint = run_dir / filename
    if not status_path.exists() or not checkpoint.exists():
        return None
    status = load_json(status_path)
    if not bool(status.get(stage, {}).get("completed")):
        return None
    return checkpoint.resolve()


def training_is_complete(run_dir: Path) -> bool:
    status_path = run_dir / "stage_status.json"
    best_path = run_dir / "best_checkpoint.json"
    final_path = run_dir / "checkpoints" / f"step_{ORIGINAL_BUDGETS['bppo']:07d}"
    if not (status_path.exists() and best_path.exists() and final_path.exists()):
        return False
    return bool(load_json(status_path).get("bppo", {}).get("completed"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen faithful BPPO configuration on the full gamma-v3 suite."
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
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_seed", type=int, default=10000)
    parser.add_argument("--reeval_episodes", type=int, default=200)
    parser.add_argument("--reeval_seed", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_missing_datasets", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dt_repo = (repo_root / args.dt_repo).resolve()
    machine_types = parse_csv(args.machine_types)
    setup_costs = parse_csv(args.setup_cost_per_machine, int)
    machine_counts = parse_csv(args.num_machines, int)
    modes = parse_csv(args.modes)
    seeds = parse_csv(args.seeds, int)
    suite_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else repo_root / "Seed runs" / "gamma_v3_bppo_faithful_full"
    )

    requested = [
        (machine_type, setup_cost, num_machines, mode)
        for machine_type in machine_types
        for setup_cost in setup_costs
        for num_machines in machine_counts
        for mode in modes
    ]
    prepared = []
    missing = []
    for machine_type, setup_cost, num_machines, mode in requested:
        env_name = environment_name(machine_type, setup_cost, num_machines)
        try:
            study = prepare_study(
                repo_root, dt_repo, machine_type, setup_cost, num_machines,
                mode, args.num_episodes, args.horizon, args.discount,
                args.dataset_seed,
            )
            prepared.append((machine_type, setup_cost, num_machines, mode, study))
        except FileNotFoundError as error:
            dataset_path, metadata_path = dataset_files(
                dt_repo, env_name, num_machines, mode, args.num_episodes)
            missing.append({
                "machine_type": machine_type,
                "setup_cost_per_machine": setup_cost,
                "num_machines": num_machines,
                "mode": mode,
                "environment": env_name,
                "dataset_path": str(dataset_path.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "missing_paths": [
                    str(path.resolve()) for path in (dataset_path, metadata_path)
                    if not path.exists()
                ],
                "reason": str(error),
            })

    suite_spec = {
        "method": "BPPO",
        "protocol": "frozen faithful full gamma-v3 benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dt_repo": str(dt_repo),
        "requested_studies": len(requested),
        "available_studies": len(prepared),
        "missing_studies": len(missing),
        "seeds": seeds,
        "discount": args.discount,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "reeval_episodes": args.reeval_episodes,
        "reeval_seed": args.reeval_seed,
        "frozen_hyperparameters": {
            "clip_ratio": FROZEN_CLIP_RATIO,
            "decay": FROZEN_DECAY,
            "learning_rate": 1e-4,
            "batch_size": 512,
            "target_update_freq": 2,
            "tau": 0.005,
            "omega": 0.9,
            "entropy_weight": 0.0,
            "clip_decay_first_steps": 200,
            "lr_decay_first_steps": 200,
            "update_old_policy_on_improvement": True,
            "bppo_eval_interval": 1,
        },
        "stage_budgets": ORIGINAL_BUDGETS,
        "networks": ORIGINAL_NETWORKS,
        "checkpoint_selection": "minimum 10-episode BPPO evaluation cost, including the initial BC policy",
        "primary_report": "selected checkpoint reevaluated for 200 episodes",
        "secondary_report": "final step-1000 checkpoint reevaluated for 200 episodes",
    }
    write_json(suite_root / "suite_spec.json", suite_spec)
    write_json(suite_root / "missing_datasets.json", {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "missing": missing,
    })
    if missing and not args.skip_missing_datasets:
        raise FileNotFoundError(
            f"{len(missing)} requested dataset(s) are missing. See "
            f"{suite_root / 'missing_datasets.json'}, or pass --skip_missing_datasets."
        )
    if args.prepare_only:
        print(
            f"[PREPARED] available={len(prepared)} missing={len(missing)} root={suite_root}",
            flush=True,
        )
        return

    suite_rows = []
    for machine_type, setup_cost, num_machines, mode, study in prepared:
        study_root = suite_root / study.env_name / f"M{num_machines}" / mode / "bppo"
        study_spec = {
            **suite_spec,
            "environment": study.env_name,
            "machine_type": machine_type,
            "setup_cost_per_machine": setup_cost,
            "num_machines": num_machines,
            "mode": mode,
            "dataset_path": str(study.dataset_path),
            "metadata_path": str(study.metadata_path),
        }
        write_json(study_root / "experiment_spec.json", study_spec)
        seed_rows = []
        for seed in seeds:
            args.seed = seed
            args.num_machines = num_machines
            run_dir = study_root / f"seed_{seed}"
            result_path = run_dir / "benchmark_result.json"
            if args.resume and result_path.exists():
                print(f"[RESUME] complete: {run_dir}", flush=True)
                seed_rows.append(load_json(result_path))
                continue
            if run_dir.exists() and any(run_dir.iterdir()) and not args.resume and not args.skip_train:
                raise FileExistsError(
                    f"Existing incomplete output requires --resume: {run_dir}")

            if not args.skip_train and not (args.resume and training_is_complete(run_dir)):
                value_checkpoint = (
                    completed_stage_checkpoint(run_dir, "value", "value_final.pt")
                    if args.resume else None)
                q_checkpoint = (
                    completed_stage_checkpoint(run_dir, "q", "q_final.pt")
                    if args.resume else None)
                bc_checkpoint = (
                    completed_stage_checkpoint(run_dir, "bc", "bc_best.pt")
                    if args.resume else None)
                run(training_command(
                    repo_root, study, run_dir, args,
                    bppo_steps=ORIGINAL_BUDGETS["bppo"],
                    clip_ratio=FROZEN_CLIP_RATIO,
                    decay=FROZEN_DECAY,
                    value_checkpoint=value_checkpoint,
                    q_checkpoint=q_checkpoint,
                    bc_checkpoint=bc_checkpoint,
                ), repo_root)

            best_path = run_dir / "best_checkpoint.json"
            final_checkpoint = run_dir / "checkpoints" / f"step_{ORIGINAL_BUDGETS['bppo']:07d}"
            if not (best_path.exists() and final_checkpoint.exists()):
                raise FileNotFoundError(f"Incomplete BPPO run: {run_dir}")
            best = load_json(best_path)
            selected_checkpoint = Path(best["checkpoint_dir"])
            selected_eval_path = run_dir / f"selected_reeval_{args.reeval_episodes}.json"
            final_eval_path = run_dir / f"final_reeval_{args.reeval_episodes}.json"
            if not (args.resume and selected_eval_path.exists()):
                run(evaluation_command(
                    repo_root, study, selected_checkpoint,
                    selected_eval_path, args), repo_root)
            if not (args.resume and final_eval_path.exists()):
                run(evaluation_command(
                    repo_root, study, final_checkpoint,
                    final_eval_path, args), repo_root)
            selected_eval = load_json(selected_eval_path)
            final_eval = load_json(final_eval_path)
            seed_result = {
                "seed": seed,
                "selected_step": int(best["best_step"]),
                "selected_checkpoint_dir": str(selected_checkpoint),
                "final_checkpoint_dir": str(final_checkpoint),
                "selected_reevaluation": selected_metrics(selected_eval),
                "final_reevaluation": selected_metrics(final_eval),
            }
            write_json(result_path, seed_result)
            seed_rows.append(seed_result)

        selected_costs = [
            float(row["selected_reevaluation"]["discounted_cost"])
            for row in seed_rows]
        final_costs = [
            float(row["final_reevaluation"]["discounted_cost"])
            for row in seed_rows]
        study_summary = {
            **study_spec,
            "per_seed": seed_rows,
            "selected_discounted_cost_across_seeds": summarize(selected_costs),
            "final_discounted_cost_across_seeds": summarize(final_costs),
        }
        write_json(study_root / "summary.json", study_summary)
        suite_rows.append(study_summary)
        print(
            f"[DONE] {study.env_name} M{num_machines} {mode}: "
            f"selected_cost={study_summary['selected_discounted_cost_across_seeds']['mean']:.6f}",
            flush=True,
        )

    write_json(suite_root / "suite_summary.json", {
        **suite_spec,
        "completed_studies": len(suite_rows),
        "studies": suite_rows,
    })


if __name__ == "__main__":
    main()
