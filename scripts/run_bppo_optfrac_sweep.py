import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fraction_tag(value):
    return f"of_{value:.2f}".replace(".", "p")


def manifest_tag(item):
    if item.get("dataset_tag"):
        return str(item["dataset_tag"])
    return fraction_tag(float(item["optimal_fraction"]))


def filter_manifest(manifest, fraction):
    if fraction is None:
        return manifest
    requested_tag = fraction_tag(fraction)
    requested_pct = round(fraction * 100)
    filtered = []
    for item in manifest:
        item_fraction = float(item["optimal_fraction"])
        item_tag = fraction_tag(item_fraction / 100.0 if item_fraction > 1.0 else item_fraction)
        if (
            abs(item_fraction - fraction) < 1e-12
            or abs(item_fraction - requested_pct) < 1e-12
            or requested_tag in str(item.get("dataset_path", ""))
            or item_tag == requested_tag
        ):
            filtered.append(item)
    if not filtered:
        raise ValueError(f"No manifest entry found for optimal_fraction={fraction}")
    return filtered


def build_parser(defaults=None):
    parser = argparse.ArgumentParser(
        description="Run BPPO training for a manifest-driven maintenance dataset sweep."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--policy_path", type=str, default=None)
    parser.add_argument("--extra_policy_paths", type=str, default=None)
    parser.add_argument("--env_name", type=str, default="mm_default")
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--is_state_norm", action="store_true")

    parser.add_argument("--v_steps", type=int, default=20000)
    parser.add_argument("--v_hidden_dim", type=int, default=256)
    parser.add_argument("--v_depth", type=int, default=2)
    parser.add_argument("--v_lr", type=float, default=1e-4)
    parser.add_argument("--v_batch_size", type=int, default=512)
    parser.add_argument("--q_bc_steps", type=int, default=20000)
    parser.add_argument("--q_hidden_dim", type=int, default=256)
    parser.add_argument("--q_depth", type=int, default=2)
    parser.add_argument("--q_lr", type=float, default=1e-4)
    parser.add_argument("--q_batch_size", type=int, default=512)
    parser.add_argument("--target_update_freq", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--bc_steps", type=int, default=10000)
    parser.add_argument("--bc_hidden_dim", type=int, default=256)
    parser.add_argument("--bc_depth", type=int, default=2)
    parser.add_argument("--bc_lr", type=float, default=1e-4)
    parser.add_argument("--bc_batch_size", type=int, default=512)
    parser.add_argument("--bppo_steps", type=int, default=5000)
    parser.add_argument("--bppo_hidden_dim", type=int, default=256)
    parser.add_argument("--bppo_depth", type=int, default=2)
    parser.add_argument("--bppo_lr", type=float, default=1e-4)
    parser.add_argument("--bppo_batch_size", type=int, default=512)
    parser.add_argument("--clip_ratio", type=float, default=0.25)
    parser.add_argument("--entropy_weight", type=float, default=0.0)
    parser.add_argument("--decay", type=float, default=0.96)
    parser.add_argument("--omega", type=float, default=0.9)
    parser.add_argument("--is_clip_decay", action="store_true")
    parser.add_argument("--is_bppo_lr_decay", action="store_true")
    parser.add_argument("--is_update_old_policy", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--dry_run", action="store_true")
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def parse_args():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()

    defaults = None
    if bootstrap_args.config:
        with open(bootstrap_args.config, "r", encoding="utf-8") as f:
            defaults = json.load(f)

    parser = build_parser(defaults=defaults)
    args = parser.parse_args()
    missing = [
        name for name in ("manifest_path", "repo_root", "output_root")
        if not getattr(args, name)
    ]
    if missing:
        parser.error("Missing required settings: " + ", ".join(f"--{name}" for name in missing))
    return args


def resolve_dataset_path(args, item):
    dataset_path = Path(item["dataset_path"])
    if dataset_path.exists():
        return dataset_path

    local_root = Path(args.dataset_root) if args.dataset_root else Path(args.repo_root) / "datasets"
    tag = manifest_tag(item)
    candidate = local_root / f"MM{item['num_machines']}" / tag / dataset_path.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve dataset path for manifest entry {item['dataset_path']}. "
        f"Checked local candidate: {candidate}"
    )


def build_command(args, item, run_dir):
    repo_root = Path(args.repo_root)
    cmd = [
        sys.executable,
        str(repo_root / "train_bppo_machine.py"),
        "--dataset_path", str(resolve_dataset_path(args, item)),
        "--save_dir", str(run_dir),
        "--seed", str(args.seed),
        "--env_name", str(args.env_name),
        "--num_machines", str(item["num_machines"]),
        "--horizon", str(args.horizon),
        "--episodes", str(args.episodes),
        "--discount", str(args.discount),
        "--device", str(args.device),
        "--v_steps", str(args.v_steps),
        "--v_hidden_dim", str(args.v_hidden_dim),
        "--v_depth", str(args.v_depth),
        "--v_lr", str(args.v_lr),
        "--v_batch_size", str(args.v_batch_size),
        "--q_bc_steps", str(args.q_bc_steps),
        "--q_hidden_dim", str(args.q_hidden_dim),
        "--q_depth", str(args.q_depth),
        "--q_lr", str(args.q_lr),
        "--q_batch_size", str(args.q_batch_size),
        "--target_update_freq", str(args.target_update_freq),
        "--tau", str(args.tau),
        "--bc_steps", str(args.bc_steps),
        "--bc_hidden_dim", str(args.bc_hidden_dim),
        "--bc_depth", str(args.bc_depth),
        "--bc_lr", str(args.bc_lr),
        "--bc_batch_size", str(args.bc_batch_size),
        "--bppo_steps", str(args.bppo_steps),
        "--bppo_hidden_dim", str(args.bppo_hidden_dim),
        "--bppo_depth", str(args.bppo_depth),
        "--bppo_lr", str(args.bppo_lr),
        "--bppo_batch_size", str(args.bppo_batch_size),
        "--clip_ratio", str(args.clip_ratio),
        "--entropy_weight", str(args.entropy_weight),
        "--decay", str(args.decay),
        "--omega", str(args.omega),
        "--eval_interval", str(args.eval_interval),
        "--checkpoint_interval", str(args.checkpoint_interval),
        "--log_interval", str(args.log_interval),
    ]
    if args.policy_path:
        cmd.extend(["--policy_path", str(args.policy_path)])
    if args.extra_policy_paths:
        cmd.extend(["--extra_policy_paths", str(args.extra_policy_paths)])
    if args.is_state_norm:
        cmd.append("--is_state_norm")
    if args.is_clip_decay:
        cmd.append("--is_clip_decay")
    if args.is_bppo_lr_decay:
        cmd.append("--is_bppo_lr_decay")
    if args.is_update_old_policy:
        cmd.append("--is_update_old_policy")
    return cmd


def main():
    args = parse_args()
    manifest = filter_manifest(load_manifest(Path(args.manifest_path)), args.fraction)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for item in manifest:
        tag = manifest_tag(item)
        run_dir = output_root / f"MM{item['num_machines']}" / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(args, item, run_dir)
        print("Running:", " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=args.repo_root, check=True)


if __name__ == "__main__":
    main()
