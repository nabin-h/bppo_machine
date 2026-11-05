"""Estimate evaluation-excluded BPPO wall time from completed run artifacts.

No training or performance artifacts are changed. Requires the original BPPO
repository and saved policies on the measurement computer; dry-run is stdlib only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relocated_path(raw, repo):
    """Resolve Windows paths saved under a different username or run root."""
    direct = Path(raw)
    if direct.is_file():
        return direct
    parts = raw.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part.lower() == repo.name.lower():
            candidate = repo.joinpath(*parts[i + 1:])
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Cannot locate original artifact: {raw}")


def evaluation_counts(config, history):
    counts = Counter(row.get("stage") for row in history)
    expected = {
        "bc": len(range(1, config["bc_steps"] + 1, config["bc_eval_interval"])),
        "bppo_initial_bc": 1,
        "bppo": len(set(range(config["bppo_eval_interval"], config["bppo_steps"] + 1,
                              config["bppo_eval_interval"])) | {config["bppo_steps"]}),
    }
    if dict(counts) != expected:
        raise ValueError(f"Evaluation history does not match full training schedule: {counts}; expected {expected}")
    return expected


def inspect_run(run, root, repo):
    config_path = run / "training_config.json"
    config = read_json(config_path)
    timing_path = run / "timing.json"
    timing = read_json(timing_path)
    seed = config["seed"]
    history_path = run / f"{seed}_eval.json"
    for key, target in (("v_steps_executed", config["v_steps"]),
                        ("q_bc_steps_executed", config["q_bc_steps"]),
                        ("bc_steps_executed", config["bc_steps"])):
        if timing.get(key) != target:
            raise ValueError(f"Not a full from-scratch training run: {key}={timing.get(key)}")
    counts = evaluation_counts(config, read_json(history_path))
    if config.get("is_state_norm"):
        raise ValueError("This script requires the original unnormalized-state runs")
    if config.get("resolved_device", config.get("device")) != "cuda":
        raise ValueError("Original run is not recorded as CUDA; inspect before extrapolation")
    bc = run / "bc_best.pt"
    if not bc.is_file():
        raise FileNotFoundError(bc)
    # BC used its training-mode network for evaluation (no dropout in this MLP).
    # Initial BPPO uses this same best BC policy; later BPPO checkpoints sample
    # policy-dependent runtime variation across the saved training history.
    candidates = list((run / "checkpoints").glob("step_*/policy.pt"))
    candidates.sort(key=lambda p: p.parent.name)
    selected = run / "checkpoints" / "best" / "policy.pt"
    bppo_paths = []
    if selected.is_file():
        bppo_paths.append(selected)
    if candidates:
        bppo_paths += [candidates[0], candidates[-1]]
    if not bppo_paths:
        result = read_json(run / "benchmark_result.json")
        for field in ("selected_checkpoint_dir", "final_checkpoint_dir"):
            try:
                bppo_paths.append(relocated_path(result[field] + "/policy.pt", repo))
            except FileNotFoundError:
                pass
    bppo_paths = list(dict.fromkeys(bppo_paths))
    if not bppo_paths:
        raise FileNotFoundError("No BPPO checkpoints: retain checkpoints/ from the original training folder")
    reference = relocated_path(config["policy_path"], repo) if config.get("policy_path") else None
    extra = []
    for item in config.get("extra_policy_paths") or []:
        extra.extend(relocated_path(p.strip(), repo) for p in item.split(",") if p.strip())
    files = [config_path, timing_path, history_path, bc, *bppo_paths]
    files += ([reference] if reference else []) + extra
    return dict(run=run, key=run.relative_to(root).as_posix(), config=config,
                recorded_seconds=float(timing["train_wall_time_sec"]), counts=counts,
                checkpoint_groups={"bc": [bc], "bppo_initial_bc": [bc], "bppo": bppo_paths},
                reference=reference, extra=extra,
                input_sha256={str(p): digest(p) for p in dict.fromkeys(files)})


def adjusted_estimate(recorded, counts, samples):
    overhead = sum(counts[s] * statistics.mean(samples[s]) for s in counts)
    low_overhead = sum(counts[s] * min(samples[s]) for s in counts)
    high_overhead = sum(counts[s] * max(samples[s]) for s in counts)
    if high_overhead >= recorded:
        raise ValueError("Measured overhead can exceed original elapsed time; check hardware and protocol")
    return dict(estimated_evaluation_seconds=overhead,
                estimated_evaluation_excluded_seconds=recorded - overhead,
                sensitivity_seconds=[recorded - high_overhead, recorded - low_overhead],
                sensitivity_note="Observed duration extrema propagated through call counts; not a confidence interval")


def measure(info, args, torch, trainer):
    import numpy as np
    from net import BernoulliPolicyMLP

    config = info["config"]
    device = torch.device(args.device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    env = trainer.make_env(config["num_machines"], config["horizon"], config["seed"], config["env_name"])
    state_dim = int(np.asarray(env.observation_space.sample()).shape[-1])
    action_dim = int(np.asarray(env.action_space.sample()).shape[-1])
    mean = np.zeros((1, state_dim), dtype=np.float32)
    std = np.ones((1, state_dim), dtype=np.float32)
    reference = trainer.load_reference_policy(str(info["reference"])) if info["reference"] else None
    extra = trainer.load_extra_reference_policies([str(p) for p in info["extra"]])
    samples, observations = {}, {}
    # Match deterministic Bernoulli thresholding in BinaryPolicyBase.select_action.
    class Agent:
        def __init__(self, network):
            self.network = network

        def select_action(self, state, is_sample=False):
            return (self.network(state).probs >= 0.5).float()

    try:
        for stage, checkpoints in info["checkpoint_groups"].items():
            durations, details = [], []
            for checkpoint in checkpoints:
                prefix = "bc" if stage != "bppo" else "bppo"
                network = BernoulliPolicyMLP(state_dim, config[f"{prefix}_hidden_dim"],
                                             config[f"{prefix}_depth"], action_dim).to(device)
                network.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                network.train()  # The original evaluator leaves policy mode unchanged.
                policy_fn = trainer.make_policy_fn(Agent(network), mean, std, device)

                def evaluate():
                    env.seed(config["eval_seed"])
                    return trainer.evaluate_policy_fn(
                        policy_fn, env, config["episodes"], discount=config["discount"],
                        reference_policy=reference, extra_reference_policies=extra,
                        env_name=config["env_name"])

                for _ in range(args.warmups):
                    evaluate()
                for repeat in range(args.repeats):
                    sync()
                    started = time.perf_counter()
                    result = evaluate()
                    sync()
                    seconds = time.perf_counter() - started
                    durations.append(seconds)
                    details.append(dict(checkpoint=str(checkpoint), repeat=repeat,
                                        seconds=seconds, discounted_cost=result["discounted_cost"]))
                print(f"  {stage}: {checkpoint.parent.name}, {statistics.mean(durations):.3f} s/call", flush=True)
            samples[stage], observations[stage] = durations, details
    finally:
        env.close()
    return dict(samples_seconds=samples, measurements=observations,
                **adjusted_estimate(info["recorded_seconds"], info["counts"], samples))


def write_summary(output, reports, excluded, expected):
    groups = defaultdict(list)
    for r in reports:
        groups[r["num_machines"]].append(r)
    rows = []
    for n, runs in sorted(groups.items()):
        hours = [r["estimated_evaluation_excluded_seconds"] / 3600 for r in runs]
        rows.append(dict(num_machines=n, runs=len(runs), mean_hours=statistics.mean(hours),
                         min_hours=min(hours), max_hours=max(hours)))
    summary = dict(status="complete" if len(reports) == expected else "partial",
                   eligible_runs=expected, measured_runs=len(reports), excluded=excluded,
                   rows=rows,
                   definition="Estimated recorded training-loop wall time minus sampled rollout-evaluation overhead; includes checkpointing, logging, and other loop overhead; not pure optimizer time",
                   caveats=["Use original hardware and software; historical machine-load differences are not recoverable.",
                            "Late/best BC checkpoints proxy unsaved early BC evaluations.",
                            "BPPO saved checkpoints sample policy-dependent runtime, not every historical evaluation.",
                            "Baselines before the training timer and final 200-episode reevaluations are not subtracted."])
    save_json(output / "summary.json", summary)
    lines = ["| Method | System | Runs | Estimated hours: mean (min-max) |", "|---|---|---:|---:|"]
    for r in rows:
        lines.append(f"| BPPO | M{r['num_machines']} | {r['runs']} | {r['mean_hours']:.3f} ({r['min_hours']:.3f}-{r['max_hours']:.3f}) |")
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    latex = [r"\begin{tabular}{lrr}", r"\hline",
             r"System & Runs & Estimated training time (hours)\\", r"\hline"]
    for r in rows:
        latex.append(f"M{r['num_machines']} & {r['runs']} & "
                     f"{r['mean_hours']:.3f} ({r['min_hours']:.3f}--{r['max_hours']:.3f})"
                     + r"\\")
    latex += [r"\hline", r"\end{tabular}"]
    (output / "table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bppo-repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--confirm-same-hardware", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2 or args.warmups < 1:
        parser.error("Use at least two repeats and one warmup")
    args.bppo_repo = args.bppo_repo.resolve()
    args.runs_root = args.runs_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.output_root == args.runs_root or args.runs_root in args.output_root.parents:
        parser.error("Output must be separate from original run artifacts")
    configs = sorted(args.runs_root.rglob("training_config.json"))
    if not configs:
        parser.error("No training_config.json files found")
    eligible, excluded, blocked = [], [], []
    for config in configs:
        run = config.parent
        try:
            eligible.append(inspect_run(run, args.runs_root, args.bppo_repo))
        except ValueError as error:
            excluded.append(dict(run=str(run), reason=str(error)))
        except (FileNotFoundError, KeyError) as error:
            blocked.append(dict(run=str(run), reason=str(error)))
    print(f"Ready: {len(eligible)}; excluded: {len(excluded)}; missing artifacts: {len(blocked)}", flush=True)
    if args.dry_run:
        print(json.dumps(dict(ready=[dict(run=i["key"], calls=i["counts"]) for i in eligible],
                              excluded=excluded, missing_artifacts=blocked), indent=2))
        return
    if blocked:
        parser.error("Missing artifacts; use --dry-run to inspect. Copy original checkpoint folders before measuring")
    if not eligible:
        parser.error("No eligible runs")
    if not args.confirm_same_hardware:
        parser.error("Run on the original training hardware and pass --confirm-same-hardware")
    sys.path.insert(0, str(args.bppo_repo))
    import torch
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("CUDA is required for these original GPU runs")
    sys.modules["custom_env"] = importlib.import_module("gamma_v3_custom_env")
    trainer = importlib.import_module("train_bppo_machine")
    props = torch.cuda.get_device_properties(torch.device(args.device))
    hardware = dict(platform=platform.platform(), cpu=platform.processor(),
                    gpu=props.name, gpu_memory_bytes=props.total_memory,
                    python=platform.python_version(), torch=torch.__version__,
                    cuda=torch.version.cuda, torch_threads=torch.get_num_threads())
    source_hashes = {name: digest(args.bppo_repo / name) for name in
                     ("train_bppo_machine.py", "evaluation.py", "gamma_v3_custom_env.py", "net.py", "binary_bppo.py")}
    source_hashes["timing_script"] = digest(Path(__file__))
    reports = []
    for index, info in enumerate(eligible, 1):
        print(f"[{index}/{len(eligible)}] {info['key']}", flush=True)
        identity = dict(input_sha256=info["input_sha256"], source_sha256=source_hashes,
                        hardware=hardware, repeats=args.repeats, warmups=args.warmups)
        destination = args.output_root / info["key"] / "evaluation_overhead.json"
        old = read_json(destination) if destination.is_file() else None
        if old and old.get("identity") == identity:
            report = old
            print("  reused matching measurement", flush=True)
        else:
            report = dict(identity=identity, measured_utc=datetime.now(timezone.utc).isoformat(),
                          run=info["key"], num_machines=info["config"]["num_machines"],
                          seed=info["config"]["seed"], calls=info["counts"],
                          recorded_seconds=info["recorded_seconds"],
                          **measure(info, args, torch, trainer))
            save_json(destination, report)
        reports.append(report)
        write_summary(args.output_root, reports, excluded, len(eligible))
    print((args.output_root / "table.md").read_text(encoding="utf-8"))
    print(f"Results: {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
