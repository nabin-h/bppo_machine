from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.gamma_v3_experiment import parse_csv, prepare_study, write_json


ORIGINAL_BUDGETS = {"v": 2_000_000, "q": 2_000_000, "bc": 500_000, "bppo": 1_000}
ORIGINAL_NETWORKS = {
    "value": {"hidden_dim": 512, "depth": 3},
    "q": {"hidden_dim": 1024, "depth": 2},
    "bc": {"hidden_dim": 1024, "depth": 2},
    "bppo": {"hidden_dim": 1024, "depth": 2},
}


def run(command: list[str], repo_root: Path) -> None:
    print("[CMD]", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


def tag(value: float) -> str:
    return format(value, "g").replace("-", "m").replace(".", "p")


def evaluation_command(
    repo_root: Path,
    study,
    checkpoint_dir: Path,
    output_json: Path,
    args,
) -> list[str]:
    return [
        sys.executable,
        str(repo_root / "evaluate_gamma_v3_bppo.py"),
        "--checkpoint_dir", str(checkpoint_dir),
        "--output_json", str(output_json),
        "--env_name", study.env_name,
        "--num_machines", str(args.num_machines),
        "--horizon", str(args.horizon),
        "--episodes", str(args.reeval_episodes),
        "--eval_seed", str(args.reeval_seed),
        "--discount", str(args.discount),
        "--policy_path", str(study.policy_path),
        "--extra_policy_paths", *[str(path) for path in study.extra_policy_paths],
        "--device", args.device,
    ]


def training_command(
    repo_root: Path,
    study,
    save_dir: Path,
    args,
    bppo_steps: int,
    clip_ratio: float,
    decay: float,
    value_checkpoint: Path | None = None,
    q_checkpoint: Path | None = None,
    bc_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "train_gamma_v3_bppo.py"),
        "--dataset_path", str(study.dataset_path),
        "--save_dir", str(save_dir),
        "--seed", str(args.seed),
        "--env_name", study.env_name,
        "--num_machines", str(args.num_machines),
        "--horizon", str(args.horizon),
        "--episodes", str(args.eval_episodes),
        "--eval_seed", str(args.eval_seed),
        "--discount", str(args.discount),
        "--reward_scale", "1.0",
        "--device", args.device,
        "--policy_path", str(study.policy_path),
        "--extra_policy_paths", *[str(path) for path in study.extra_policy_paths],
        "--v_steps", str(ORIGINAL_BUDGETS["v"]),
        "--v_hidden_dim", "512", "--v_depth", "3", "--v_lr", "0.0001", "--v_batch_size", "512",
        "--q_bc_steps", str(ORIGINAL_BUDGETS["q"]),
        "--q_hidden_dim", "1024", "--q_depth", "2", "--q_lr", "0.0001", "--q_batch_size", "512",
        "--target_update_freq", "2", "--tau", "0.005",
        "--bc_steps", str(ORIGINAL_BUDGETS["bc"]),
        "--bc_hidden_dim", "1024", "--bc_depth", "2", "--bc_lr", "0.0001", "--bc_batch_size", "512",
        "--bppo_steps", str(bppo_steps),
        "--bppo_hidden_dim", "1024", "--bppo_depth", "2", "--bppo_lr", "0.0001", "--bppo_batch_size", "512",
        "--clip_ratio", str(clip_ratio), "--entropy_weight", "0.0",
        "--decay", str(decay), "--omega", "0.9",
        "--is_clip_decay", "--is_bppo_lr_decay", "--is_update_old_policy",
        "--bc_eval_interval", "2000", "--bppo_eval_interval", "1",
        "--checkpoint_interval", "50", "--log_interval", "100",
    ]
    if value_checkpoint is not None:
        command.extend(["--value_checkpoint", str(value_checkpoint)])
    if q_checkpoint is not None:
        command.extend(["--q_checkpoint", str(q_checkpoint)])
    if bc_checkpoint is not None:
        command.extend(["--bc_checkpoint", str(bc_checkpoint)])
    return command


def selected_metrics(payload: dict) -> dict:
    keys = (
        "discounted_cost", "discounted_cost_std", "return", "return_std",
        "corrections", "corrections_std", "raw_invalid_action_rate",
        "forced_failure_corrections", "forced_failure_correction_rate",
        "suppressed_zero_state_corrections", "suppressed_zero_state_correction_rate",
        "other_corrections", "other_correction_rate", "action_activation_rate",
    )
    return {key: payload[key] for key in keys if key in payload}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faithful BPPO clip/decay tuning on one representative gamma-v3 task."
    )
    parser.add_argument("--dt_repo", default="../dt_machine")
    parser.add_argument("--machine_type", choices=("homogeneous", "heterogeneous"), default="heterogeneous")
    parser.add_argument("--setup_cost_per_machine", type=int, default=25)
    parser.add_argument("--num_machines", type=int, default=20)
    parser.add_argument("--mode", choices=("plausible", "threshold"), default="plausible")
    parser.add_argument("--num_episodes", type=int, default=20000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_seed", type=int, default=10000)
    parser.add_argument("--reeval_episodes", type=int, default=200)
    parser.add_argument("--reeval_seed", type=int, default=20000)
    parser.add_argument("--clip_ratios", default="0.1,0.25")
    parser.add_argument("--decays", default="0.94,0.96,0.98")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dt_repo = (repo_root / args.dt_repo).resolve()
    study = prepare_study(
        repo_root, dt_repo, args.machine_type, args.setup_cost_per_machine,
        args.num_machines, args.mode, args.num_episodes, args.horizon,
        args.discount, args.dataset_seed,
    )
    clip_ratios = parse_csv(args.clip_ratios, float)
    decays = parse_csv(args.decays, float)
    output_root = (
        Path(args.output_root).resolve() if args.output_root else
        repo_root / "Seed runs" / "gamma_v3_bppo_faithful_tuning" / study.env_name /
        f"M{args.num_machines}" / args.mode / f"seed_{args.seed}"
    )
    shared_dir = output_root / "shared_stages"
    spec = {
        "method": "BPPO",
        "purpose": "faithful clip-ratio and decay tuning before the full gamma-v3 comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(study.dataset_path),
        "metadata_path": str(study.metadata_path),
        "environment": study.env_name,
        "machine_type": args.machine_type,
        "setup_cost_per_machine": args.setup_cost_per_machine,
        "num_machines": args.num_machines,
        "mode": args.mode,
        "discount": args.discount,
        "seed": args.seed,
        "clip_ratios": clip_ratios,
        "decays": decays,
        "stage_budgets": ORIGINAL_BUDGETS,
        "networks": ORIGINAL_NETWORKS,
        "fixed_hyperparameters": {
            "learning_rate": 1e-4, "batch_size": 512, "target_update_freq": 2,
            "tau": 0.005, "omega": 0.9, "entropy_weight": 0.0,
            "clip_decay_first_steps": 200, "lr_decay_first_steps": 200,
            "update_old_policy_on_improvement": True, "bppo_eval_interval": 1,
        },
        "necessary_adaptations": [
            "factorized Bernoulli policy for binary multi-action decisions",
            "linear Q/V outputs because reward is negative cost",
            "gamma fixed to the benchmark value 0.95",
            "environment feasibility projection retained and diagnosed by cause",
        ],
    }
    write_json(output_root / "tuning_spec.json", spec)
    print(f"[PLAN] {len(clip_ratios) * len(decays)} configurations -> {output_root}", flush=True)
    if args.prepare_only:
        return

    value_path = shared_dir / "value_final.pt"
    q_path = shared_dir / "q_final.pt"
    bc_path = shared_dir / "bc_best.pt"
    shared_complete = shared_dir / "timing.json"
    if not (shared_complete.exists() and value_path.exists() and q_path.exists() and bc_path.exists()):
        if args.skip_train:
            raise FileNotFoundError("Shared value/Q/BC checkpoints are incomplete and --skip_train was set.")
        run(training_command(
            repo_root, study, shared_dir, args, bppo_steps=0,
            clip_ratio=0.25, decay=0.96,
            value_checkpoint=value_path if value_path.exists() else None,
            q_checkpoint=q_path if q_path.exists() else None,
            # A BC best file is written before the BC stage is complete. Reuse
            # it only when the end-of-run timing marker proves stage completion.
            bc_checkpoint=bc_path if shared_complete.exists() and bc_path.exists() else None,
        ), repo_root)

    rows = []
    for clip_ratio in clip_ratios:
        for decay in decays:
            run_dir = output_root / f"clip_{tag(clip_ratio)}_decay_{tag(decay)}"
            result_path = run_dir / "tuning_result.json"
            if args.resume and result_path.exists():
                print(f"[RESUME] complete: {run_dir.name}", flush=True)
                rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            best_payload_path = run_dir / "best_checkpoint.json"
            final_checkpoint = run_dir / "checkpoints" / f"step_{ORIGINAL_BUDGETS['bppo']:07d}"
            training_complete = best_payload_path.exists() and final_checkpoint.exists()
            if not args.skip_train and not (args.resume and training_complete):
                run(training_command(
                    repo_root, study, run_dir, args, bppo_steps=ORIGINAL_BUDGETS["bppo"],
                    clip_ratio=clip_ratio, decay=decay,
                    value_checkpoint=value_path, q_checkpoint=q_path, bc_checkpoint=bc_path,
                ), repo_root)

            if not best_payload_path.exists():
                raise FileNotFoundError(f"Missing completed BPPO run: {best_payload_path}")
            best_payload = json.loads(best_payload_path.read_text(encoding="utf-8"))
            selected_checkpoint = Path(best_payload["checkpoint_dir"])
            selected_eval_path = run_dir / f"selected_reeval_{args.reeval_episodes}.json"
            final_eval_path = run_dir / f"final_reeval_{args.reeval_episodes}.json"
            run(evaluation_command(repo_root, study, selected_checkpoint, selected_eval_path, args), repo_root)
            run(evaluation_command(repo_root, study, final_checkpoint, final_eval_path, args), repo_root)
            selected_eval = json.loads(selected_eval_path.read_text(encoding="utf-8"))
            final_eval = json.loads(final_eval_path.read_text(encoding="utf-8"))
            row = {
                "clip_ratio": clip_ratio,
                "decay": decay,
                "selected_step": int(best_payload["best_step"]),
                "selected_checkpoint_dir": str(selected_checkpoint),
                "final_checkpoint_dir": str(final_checkpoint),
                "selected_reevaluation": selected_metrics(selected_eval),
                "final_reevaluation": selected_metrics(final_eval),
            }
            write_json(result_path, row)
            rows.append(row)

    rows.sort(key=lambda row: float(row["selected_reevaluation"]["discounted_cost"]))
    summary = {
        **spec,
        "selection_metric": "selected-checkpoint held-out discounted cost",
        "reevaluation_episodes": args.reeval_episodes,
        "reevaluation_seed": args.reeval_seed,
        "recommended": rows[0] if rows else None,
        "ranked_results": rows,
    }
    write_json(output_root / "tuning_summary.json", summary)
    if rows:
        best = rows[0]
        print(
            f"[DONE] best clip={best['clip_ratio']} decay={best['decay']} "
            f"cost={best['selected_reevaluation']['discounted_cost']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
