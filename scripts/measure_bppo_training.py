"""Measure representative BPPO optimization without editing the BPPO trainer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


BUDGETS = {"value": 2_000_000, "q": 2_000_000, "bc": 500_000, "bppo": 1_000}


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


class UpdateTimer:
    def __init__(self, synchronize, progress, clock=time.perf_counter):
        self.synchronize = synchronize
        self.progress = progress
        self.clock = clock
        self.seconds = {stage: 0. for stage in BUDGETS}
        self.counts = {stage: 0 for stage in BUDGETS}
        self.evaluation_seconds = 0.
        self.evaluation_calls = 0
        self.originals = []

    def wrap_update(self, cls, stage):
        original = cls.update

        @functools.wraps(original)
        def measured(instance, *args, **kwargs):
            self.synchronize()
            started = self.clock()
            result = original(instance, *args, **kwargs)
            self.synchronize()
            self.seconds[stage] += self.clock() - started
            self.counts[stage] += 1
            # Progress serialization is outside the optimization timer.
            if self.counts[stage] % 10000 == 0 or self.counts[stage] == BUDGETS[stage]:
                self.progress()
            if self.counts[stage] == BUDGETS[stage]:
                print(f"[TIMING] {stage}: {self.seconds[stage] / 3600:.4f} hours", flush=True)
            return result

        inherited = "update" not in cls.__dict__
        self.originals.append((cls, "update", original, inherited))
        cls.update = measured

    def wrap_evaluation(self, module):
        original = module.evaluate_policy_fn

        @functools.wraps(original)
        def measured(*args, **kwargs):
            self.synchronize()
            started = self.clock()
            result = original(*args, **kwargs)
            self.synchronize()
            self.evaluation_seconds += self.clock() - started
            self.evaluation_calls += 1
            return result

        self.originals.append((module, "evaluate_policy_fn", original, False))
        module.evaluate_policy_fn = measured

    def restore(self):
        for owner, name, original, inherited in reversed(self.originals):
            if inherited:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        self.originals.clear()

    def snapshot(self):
        total = sum(self.seconds.values())
        return dict(optimization_seconds=total, optimization_hours=total / 3600,
                    stages={s: dict(updates=self.counts[s], expected_updates=BUDGETS[s],
                                    seconds=self.seconds[s], hours=self.seconds[s] / 3600)
                            for s in BUDGETS},
                    evaluation_seconds=self.evaluation_seconds,
                    evaluation_calls=self.evaluation_calls,
                    all_stages_complete=self.counts == BUDGETS)


def paths_for(args, n):
    environment = f"gamma_v3_cs{25 * n}"
    dataset = (args.dt_repo / f"datasets_{environment}" / f"M{n}"
               / "plausible_mix_20000" / f"traj_{environment}_M{n}_plausible_20000.pkl")
    output = args.output_root / environment / f"M{n}" / f"seed_{args.seed}"
    return environment, dataset, output


def validate_dataset(args, n):
    environment, dataset, _ = paths_for(args, n)
    metadata_path = dataset.parent / "dataset_metadata.json"
    if not dataset.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing existing dataset or metadata: {dataset}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, expected in (("num_machines", n), ("num_episodes", 20000), ("horizon", 100)):
        if metadata.get(key) != expected:
            raise ValueError(f"Unexpected dataset {key}: {metadata.get(key)}")
    if abs(float(metadata.get("discount", -1)) - .95) > 1e-12:
        raise ValueError("Unexpected dataset discount")
    env = metadata.get("environment", {})
    if env.get("env_name") != environment or float(env.get("setup_cost", -1)) != 25 * n:
        raise ValueError("Unexpected dataset environment or setup cost")


def trainer_arguments(args, n, dataset, output):
    environment = f"gamma_v3_cs{25 * n}"
    return ["train_gamma_v3_bppo.py", "--dataset_path", str(dataset), "--save_dir", str(output),
            "--seed", str(args.seed), "--env_name", environment, "--num_machines", str(n),
            "--horizon", "100", "--discount", "0.95", "--reward_scale", "1.0",
            "--device", args.device, "--episodes", "10", "--eval_seed", "10000",
            "--v_steps", str(BUDGETS["value"]), "--v_hidden_dim", "512", "--v_depth", "3",
            "--v_lr", "0.0001", "--v_batch_size", "512",
            "--q_bc_steps", str(BUDGETS["q"]), "--q_hidden_dim", "1024", "--q_depth", "2",
            "--q_lr", "0.0001", "--q_batch_size", "512", "--target_update_freq", "2", "--tau", ".005",
            "--bc_steps", str(BUDGETS["bc"]), "--bc_hidden_dim", "1024", "--bc_depth", "2",
            "--bc_lr", "0.0001", "--bc_batch_size", "512",
            "--bppo_steps", str(BUDGETS["bppo"]), "--bppo_hidden_dim", "1024", "--bppo_depth", "2",
            "--bppo_lr", "0.0001", "--bppo_batch_size", "512", "--clip_ratio", ".25",
            "--entropy_weight", "0", "--decay", ".96", "--omega", ".9",
            "--is_clip_decay", "--is_bppo_lr_decay", "--is_update_old_policy",
            "--bc_eval_interval", "2000", "--bppo_eval_interval", "20",
            "--checkpoint_interval", "50", "--log_interval", "100"]


def run_worker(args, n):
    validate_dataset(args, n)
    environment, dataset, output = paths_for(args, n)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Fresh timing requires an empty run directory: {output}")
    sys.path.insert(0, str(args.bppo_repo))
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not substitute a CPU run for the GPU timing")
    device = torch.device(args.device)
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    # Match the existing gamma-v3 entry point before importing the trainer.
    sys.modules["custom_env"] = importlib.import_module("gamma_v3_custom_env")
    trainer = importlib.import_module("train_bppo_machine")
    hashes = {name: hashlib.sha256((args.bppo_repo / name).read_bytes()).hexdigest()
              for name in ("train_bppo_machine.py", "binary_bppo.py", "critic.py", "net.py",
                           "gamma_v3_custom_env.py")}
    hardware = dict(platform=platform.platform(), cpu=platform.processor(),
                    torch_version=torch.__version__, python_version=platform.python_version(),
                    device=str(device), torch_threads=torch.get_num_threads())
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        hardware.update(gpu_name=props.name, gpu_memory_bytes=props.total_memory,
                        cuda_version=torch.version.cuda)
    report = dict(status="running", environment=environment, num_machines=n, seed=args.seed,
                  created_utc=datetime.now(timezone.utc).isoformat(), hardware=hardware,
                  source_sha256=hashes, dataset_path=str(dataset), dataset_bytes=dataset.stat().st_size,
                  timing_definition="Synchronized update calls, including minibatch sampling and all optimizer stages; excludes evaluation, checkpointing, initialization, and dataset loading",
                  timing_overhead="CUDA synchronization at both boundaries of each update; Python wrapper overhead excluded",
                  representative_measurement=True,
                  note="Existing performance results are untouched; no pretrained checkpoints are reused")
    timer = UpdateTimer(synchronize, lambda: save_json(output / "optimization_timing.json",
                                                      dict(report, **timer.snapshot())))
    old_argv = sys.argv
    started = time.perf_counter()
    try:
        timer.wrap_update(trainer.ValueLearner, "value")
        timer.wrap_update(trainer.QSarsaLearner, "q")
        timer.wrap_update(trainer.BehaviorCloning, "bc")
        timer.wrap_update(trainer.BehaviorProximalPolicyOptimization, "bppo")
        timer.wrap_evaluation(trainer)
        save_json(output / "optimization_timing.json", dict(report, **timer.snapshot()))
        sys.argv = trainer_arguments(args, n, dataset, output)
        trainer.main()
        if timer.counts != BUDGETS:
            raise RuntimeError(f"Incomplete or reused stages: {timer.counts}")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=str(error))
        raise
    finally:
        synchronize()
        report["instrumented_process_seconds"] = time.perf_counter() - started
        save_json(output / "optimization_timing.json", dict(report, **timer.snapshot()))
        timer.restore()
        sys.argv = old_argv
    print(f"[COMPLETE] M{n}: optimization={sum(timer.seconds.values()) / 3600:.4f} hours", flush=True)


def main():
    script = Path(__file__).resolve()
    in_repository = script.parent.name == "scripts" and (script.parent.parent / "train_bppo_machine.py").is_file()
    default_repo = script.parent.parent if in_repository else script.parents[3] / "BPPO_machine"
    default_output = (default_repo / "Seed runs" / "bppo_optimization_timing"
                      if in_repository else script.parent / "results")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bppo-repo", type=Path, default=default_repo)
    parser.add_argument("--dt-repo", type=Path, default=default_repo.parent / "dt_machine")
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--components", default="20,40")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.bppo_repo, args.dt_repo, args.output_root = (
        p.resolve() for p in (args.bppo_repo, args.dt_repo, args.output_root))
    sizes = [int(x) for x in args.components.split(",")]
    if not sizes or len(sizes) != len(set(sizes)) or any(n not in (20, 40) for n in sizes):
        parser.error("Choose M20 and/or M40 once each")
    for name in ("train_bppo_machine.py", "gamma_v3_custom_env.py", "binary_bppo.py", "critic.py"):
        if not (args.bppo_repo / name).is_file():
            raise FileNotFoundError(args.bppo_repo / name)
    if args.worker:
        if len(sizes) != 1:
            parser.error("Worker requires one size")
        run_worker(args, sizes[0])
        return
    reports = []
    for n in sizes:
        environment, dataset, output = paths_for(args, n)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--components", str(n),
                   "--bppo-repo", str(args.bppo_repo), "--dt-repo", str(args.dt_repo),
                   "--output-root", str(args.output_root), "--seed", str(args.seed), "--device", args.device]
        print(f"[PLAN] {environment}; all stages from scratch; dataset exists={dataset.is_file()}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        validate_dataset(args, n)
        report_path = output / "optimization_timing.json"
        if report_path.exists():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if previous.get("status") == "complete" and previous.get("all_stages_complete"):
                print(f"[SKIP] Completed representative timing: {report_path}", flush=True)
                reports.append(previous)
                continue
            raise FileExistsError(f"Partial timing run cannot be used as a fresh measurement. Use a new --output-root: {output}")
        subprocess.run(command, cwd=args.bppo_repo, check=True)
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        save_json(args.output_root / "representative_timing_summary.json", dict(runs=reports))


if __name__ == "__main__":
    main()
