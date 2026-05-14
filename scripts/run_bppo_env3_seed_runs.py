import argparse
import subprocess
import sys
from pathlib import Path


def build_job(repo_root: Path, config_name: str, output_root: Path, seeds: str, reeval_episodes: int):
    return [
        sys.executable,
        str(repo_root / "scripts" / "run_bppo_multiseed_pipeline.py"),
        "--config",
        str(repo_root / "configs" / config_name),
        "--output_root",
        str(output_root),
        "--seeds",
        seeds,
        "--reeval_episodes",
        str(reeval_episodes),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run fixed-budget env3 BPPO multiseed jobs for M30, M20, then M10."
    )
    parser.add_argument("--base_output_dir", type=str, default="Seed runs")
    parser.add_argument("--seeds", type=str, default="123,234,345")
    parser.add_argument("--reeval_episodes", type=int, default=1000)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_reeval", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    base_output_dir = Path(args.base_output_dir)
    jobs = [
        (
            "mm30_env3_567_r20_seedruns.json",
            base_output_dir / "M30 runs" / "runs_mm30_env3_567_r20_bppo_seedruns",
        ),
        (
            "mm20_env3_567_r20_seedruns.json",
            base_output_dir / "M20 runs" / "runs_mm20_env3_567_r20_bppo_seedruns",
        ),
        (
            "m10s10_env3_567_r20_seedruns.json",
            base_output_dir / "M10 runs" / "runs_m10s10_env3_567_r20_bppo_seedruns",
        ),
    ]

    for config_name, output_root in jobs:
        print(f"[QUEUE] starting {config_name}")
        print(f"[QUEUE] output_root={output_root}")
        cmd = build_job(
            repo_root=repo_root,
            config_name=config_name,
            output_root=output_root,
            seeds=args.seeds,
            reeval_episodes=args.reeval_episodes,
        )
        if args.skip_train:
            cmd.append("--skip_train")
        if args.skip_reeval:
            cmd.append("--skip_reeval")
        subprocess.run(cmd, cwd=repo_root, check=True)
        print(f"[QUEUE] finished {config_name}")


if __name__ == "__main__":
    main()
