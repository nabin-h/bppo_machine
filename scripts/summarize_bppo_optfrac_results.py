import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_timing(run_dir: Path):
    timing_path = run_dir / "timing.json"
    if not timing_path.exists():
        return {}
    return load_json(timing_path)


def parse_run_tag(tag: str):
    if tag.startswith("of_"):
        return float(tag.split("_", 1)[1].replace("p", ".")), tag
    return None, tag


def main():
    parser = argparse.ArgumentParser(description="Summarize BPPO maintenance runs.")
    parser.add_argument("--runs_root", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    summary = []

    for eval_path in sorted(runs_root.glob("MM*/*/*_eval.json")):
        if eval_path.name == "baseline_eval.json":
            continue
        rows = load_json(eval_path)
        if not isinstance(rows, list) or not rows:
            continue
        best = max(rows, key=lambda row: row["return"])
        last = rows[-1]
        run_dir = eval_path.parent
        timing = load_timing(run_dir)
        optimal_fraction, dataset_tag = parse_run_tag(run_dir.name)
        summary.append({
            "machine_group": run_dir.parent.name,
            "optimal_fraction": optimal_fraction,
            "dataset_tag": dataset_tag,
            "run_dir": str(run_dir.resolve()),
            "best_step": best["step"],
            "best_return": best["return"],
            "best_cost": best.get("cost"),
            "best_full_match_pct": best.get("full_match_pct"),
            "best_corrections": best.get("corrections"),
            "final_step": last["step"],
            "final_return": last["return"],
            "final_cost": last.get("cost"),
            "final_full_match_pct": last.get("full_match_pct"),
            "final_corrections": last.get("corrections"),
            "train_wall_time_sec": timing.get("train_wall_time_sec"),
            "v_steps": timing.get("v_steps"),
            "q_bc_steps": timing.get("q_bc_steps"),
            "bc_steps": timing.get("bc_steps"),
            "bppo_steps": timing.get("bppo_steps"),
        })

    summary.sort(
        key=lambda row: (
            row["machine_group"],
            0 if row["optimal_fraction"] is not None else 1,
            row["optimal_fraction"] if row["optimal_fraction"] is not None else row["dataset_tag"],
        )
    )
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
