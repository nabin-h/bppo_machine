import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_threshold_reference(runs_root, metric_name):
    runs_root = Path(runs_root)
    for baseline_path in sorted(runs_root.glob("MM*/*/baseline_eval.json")):
        payload = load_json(baseline_path)
        threshold = payload.get("threshold")
        if not isinstance(threshold, dict):
            continue
        value = threshold.get(metric_name)
        if value is not None:
            return float(value)
    return None


def format_label(tag):
    if tag.startswith("of_"):
        value = float(tag.split("_", 1)[1].replace("p", "."))
        pct = value * 100.0
        if abs(pct - round(pct)) < 1e-9:
            return f"{int(round(pct))}%"
        return f"{pct:g}%"
    return tag.replace("_", " ")


def load_series(runs_root):
    series_by_tag = {}
    runs_root = Path(runs_root)
    for eval_path in sorted(runs_root.glob("MM*/*/*_eval.json")):
        rows = load_json(eval_path)
        if not isinstance(rows, list) or (rows and not isinstance(rows[0], dict)):
            continue
        series_by_tag[eval_path.parent.name] = rows
    return series_by_tag


def plot_metric(series_by_tag, metric_key, std_key, ylabel, title, output_path, reference_cost=None):
    plt.figure(figsize=(5, 3))
    max_x = 0
    all_y = []

    for tag in sorted(series_by_tag.keys()):
        rows = series_by_tag[tag]
        x = np.asarray([row["step"] for row in rows], dtype=np.float64)
        y = np.asarray([row.get(metric_key, np.nan) for row in rows], dtype=np.float64)
        plt.plot(x, y, label=format_label(tag), linewidth=1.2)
        if std_key is not None:
            yerr = np.asarray([row.get(std_key, np.nan) for row in rows], dtype=np.float64)
            if np.isfinite(yerr).any():
                plt.fill_between(x, y - yerr, y + yerr, alpha=0.12)
        max_x = max(max_x, float(np.max(x)))
        all_y.extend(y[np.isfinite(y)].tolist())

    plt.xlabel("Training Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.minorticks_on()
    if metric_key == "cost" and reference_cost is not None:
        plt.axhline(y=reference_cost, color="g", linestyle="--", label="R*", alpha=0.7, linewidth=1.0)
    if all_y:
        lower = min(all_y)
        if metric_key == "cost":
            lower = 500 * np.floor(lower / 500.0)
        else:
            lower = max(0.0, lower)
        plt.ylim(bottom=lower)
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, which="major", linestyle="-", alpha=0.3)
    plt.grid(True, which="minor", linestyle="-", alpha=0.08)
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot BPPO maintenance run curves.")
    parser.add_argument("--runs_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--reference_cost", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    series_by_tag = load_series(args.runs_root)
    if not series_by_tag:
        raise FileNotFoundError(f"No *_eval.json files found under {args.runs_root}")

    reference_cost = args.reference_cost
    if reference_cost is None:
        reference_cost = infer_threshold_reference(args.runs_root, "discounted_cost")

    plot_metric(series_by_tag, "cost", "cost_std", "Discounted Cost", "BPPO Cost over Training Steps",
                output_dir / "bppo_cost.png", reference_cost=reference_cost)
    plot_metric(series_by_tag, "full_match_pct", "full_match_pct_std", "Full Match (%)", "BPPO Full Match over Training Steps",
                output_dir / "bppo_full_match.png")
    plot_metric(series_by_tag, "corrections", "corrections_std", "Corrections", "BPPO Corrections over Training Steps",
                output_dir / "bppo_corrections.png")
    plot_metric(series_by_tag, "action_ones_per_step", "action_ones_per_step_std", "Action Ones per Step", "BPPO Action Rate over Training Steps",
                output_dir / "bppo_action_ones.png")


if __name__ == "__main__":
    main()
