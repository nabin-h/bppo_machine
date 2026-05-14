import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint_dir(raw_path: str, run_dir: Path, repo_root: Path) -> Path:
    checkpoint_dir = Path(raw_path)
    if checkpoint_dir.is_absolute():
        return checkpoint_dir

    repo_candidate = repo_root / checkpoint_dir
    if repo_candidate.exists():
        return repo_candidate.resolve()

    run_candidate = run_dir / checkpoint_dir
    if run_candidate.exists():
        return run_candidate.resolve()

    return repo_candidate.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Resolve the selected BPPO checkpoint from the training-produced best checkpoint file."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).resolve()
    best_path = run_dir / "best_checkpoint.json"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint file: {best_path}")

    best = load_json(best_path)
    raw_checkpoint_dir = best.get("checkpoint_dir")
    if not raw_checkpoint_dir:
        raise ValueError(f"No checkpoint_dir recorded in {best_path}")

    checkpoint_dir = resolve_checkpoint_dir(str(raw_checkpoint_dir), run_dir, repo_root)
    selection = {
        "selection_rule": (
            "Use the best BPPO checkpoint already selected during training by "
            "minimum discounted evaluation cost."
        ),
        "selected_step": int(best["best_step"]),
        "selected_discounted_cost": float(best["best_cost"]),
        "selected_discounted_return": float(best["best_return"]),
        "selected_checkpoint_dir": str(checkpoint_dir),
        "source_best_checkpoint_json": str(best_path),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)


if __name__ == "__main__":
    main()
