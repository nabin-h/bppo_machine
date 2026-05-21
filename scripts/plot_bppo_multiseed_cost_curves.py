import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_selected_step(summary: dict, seed: int):
    for row in summary.get("per_seed", []):
        if int(row["seed"]) == int(seed):
            return int(row["selected_step"])
    return None


def plot_cost_curves(output_root: Path, summary: dict, output_dir: Path, run_tag: str):
    seeds = [int(seed) for seed in summary["seeds"]]
    fig, axes = plt.subplots(1, len(seeds), figsize=(5.2 * len(seeds), 4.2), sharey=True)
    if len(seeds) == 1:
        axes = [axes]

    for ax, seed in zip(axes, seeds):
        run_dir = output_root / f"seed_{seed}" / "MM40" / run_tag
        eval_rows = [row for row in load_json(run_dir / f"{seed}_eval.json") if row.get("stage") == "bppo"]
        x = np.asarray([int(row["step"]) for row in eval_rows], dtype=np.float64)
        y = np.asarray([float(row["cost"]) for row in eval_rows], dtype=np.float64)
        ax.plot(x, y, color="#1f77b4", linewidth=1.7, label="BPPO")

        selected_step = find_selected_step(summary, seed)
        if selected_step is not None:
            for row in eval_rows:
                if int(row["step"]) == selected_step:
                    ax.scatter([selected_step], [float(row["cost"])], s=36, color="#1f77b4", edgecolor="black", linewidth=0.4, zorder=5, label="Selected checkpoint")
                    break

        ax.set_title(f"seed {seed}")
        ax.set_xlabel("BPPO Step")
        ax.grid(True, which="major", alpha=0.28)
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.08)

    axes[0].set_ylabel("Discounted Cost")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("BPPO multiseed discounted-cost curves", y=1.08, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "bppo_multiseed_discounted_cost_by_seed.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot BPPO multiseed discounted-cost curves by seed.")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default="mix_t567_e_r20")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    summary_json = Path(args.summary_json) if args.summary_json else output_root / "summary_multiseed.json"
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(summary_json)
    plot_cost_curves(output_root, summary, output_dir, args.run_tag)
    print(output_dir)


if __name__ == "__main__":
    main()
