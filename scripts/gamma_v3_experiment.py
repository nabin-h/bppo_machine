from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path


ENVIRONMENTS = {
    ("homogeneous", 5, 20): "gamma_v3_cs100",
    ("homogeneous", 5, 40): "gamma_v3_cs200",
    ("homogeneous", 25, 20): "gamma_v3_cs500",
    ("homogeneous", 25, 40): "gamma_v3_cs1000",
    ("heterogeneous", 5, 20): "gamma_v3_hetero_cs5n_m20",
    ("heterogeneous", 5, 40): "gamma_v3_hetero_cs5n_m40",
    ("heterogeneous", 25, 20): "gamma_v3_hetero_cs25n_m20",
    ("heterogeneous", 25, 40): "gamma_v3_hetero_cs25n_m40",
}


@dataclass(frozen=True)
class StudyFiles:
    env_name: str
    dataset_path: Path
    metadata_path: Path
    policy_path: Path
    extra_policy_paths: tuple[Path, ...]
    output_tag: str


def parse_csv(raw: str, cast=str) -> list:
    values = [cast(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    return values


def environment_name(machine_type: str, setup_cost_per_machine: int, num_machines: int) -> str:
    key = (machine_type, setup_cost_per_machine, num_machines)
    if key not in ENVIRONMENTS:
        raise ValueError(f"Unsupported gamma-v3 study: {key}")
    return ENVIRONMENTS[key]


def dataset_files(
    dt_repo: Path,
    env_name: str,
    num_machines: int,
    mode: str,
    num_episodes: int,
) -> tuple[Path, Path]:
    if mode == "plausible":
        directory = (
            dt_repo / f"datasets_{env_name}" / f"M{num_machines}"
            / f"plausible_mix_{num_episodes}"
        )
        filename = f"traj_{env_name}_M{num_machines}_plausible_{num_episodes}.pkl"
    elif mode == "threshold":
        directory = (
            dt_repo / f"datasets_{env_name}_threshold_only" / f"M{num_machines}"
            / "t5_to_t9_equal"
        )
        filename = f"traj_{env_name}_M{num_machines}_threshold_only_{num_episodes}.pkl"
    else:
        raise ValueError("mode must be 'plausible' or 'threshold'")
    return directory / filename, directory / "dataset_metadata.json"


def validate_metadata(
    metadata: dict,
    env_name: str,
    num_machines: int,
    num_episodes: int,
    horizon: int,
    discount: float,
    dataset_seed: int,
    setup_cost_per_machine: int,
) -> None:
    expected = {
        "num_machines": num_machines,
        "num_episodes": num_episodes,
        "horizon": horizon,
        "dataset_seed": dataset_seed,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Dataset metadata {key}={metadata.get(key)!r}, expected {value!r}")
    if abs(float(metadata.get("discount", -1.0)) - discount) > 1e-12:
        raise ValueError("Dataset discount does not match the requested discount.")
    environment = metadata.get("environment", {})
    if environment.get("env_name") != env_name:
        raise ValueError("Dataset environment name does not match the requested environment.")
    expected_setup_cost = float(setup_cost_per_machine * num_machines)
    if abs(float(environment.get("setup_cost", -1.0)) - expected_setup_cost) > 1e-12:
        raise ValueError("Dataset setup cost does not match Cs = cN.")
    reward_definition = str(metadata.get("reward_definition", "reward = -cost")).lower()
    if "-cost" not in reward_definition and "negative_cost" not in reward_definition:
        raise ValueError("The comparison pipeline requires trajectories with reward = -cost.")


def _policy_payload(label: str, env_name: str, num_machines: int) -> dict:
    if "_o" in label:
        trigger, opportunity = (int(value) for value in label[1:].split("_o"))
        return {
            "policy_type": "opportunity_two_threshold",
            "trigger": trigger,
            "opportunity_threshold": opportunity,
            "env_name": env_name,
            "num_machines": num_machines,
        }
    return {
        "policy_type": "component_threshold",
        "threshold": int(label[1:]),
        "env_name": env_name,
        "num_machines": num_machines,
    }


def prepare_study(
    repo_root: Path,
    dt_repo: Path,
    machine_type: str,
    setup_cost_per_machine: int,
    num_machines: int,
    mode: str,
    num_episodes: int,
    horizon: int,
    discount: float,
    dataset_seed: int,
) -> StudyFiles:
    env_name = environment_name(machine_type, setup_cost_per_machine, num_machines)
    dataset_path, metadata_path = dataset_files(
        dt_repo, env_name, num_machines, mode, num_episodes
    )
    if not dataset_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "The exact DT dataset and metadata are required. Missing: "
            f"{dataset_path if not dataset_path.exists() else metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(
        metadata,
        env_name,
        num_machines,
        num_episodes,
        horizon,
        discount,
        dataset_seed,
        setup_cost_per_machine,
    )

    labels = set(metadata.get("behavior_counts", {}).keys())
    labels.update({"t8_o5", "t9_o5", "t9_o6"})
    if machine_type == "homogeneous":
        labels.add("t9_o4")
    policy_dir = repo_root / "policies" / "gamma_v3_comparison"
    policy_dir.mkdir(parents=True, exist_ok=True)
    policy_paths = {}
    for label in sorted(labels):
        path = policy_dir / f"M{num_machines}_{env_name}_{label}.pkl"
        with path.open("wb") as handle:
            pickle.dump(
                _policy_payload(label, env_name, num_machines),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        policy_paths[label] = path

    primary_label = "t8_o5"
    extras = tuple(path for label, path in policy_paths.items() if label != primary_label)
    return StudyFiles(
        env_name=env_name,
        dataset_path=dataset_path.resolve(),
        metadata_path=metadata_path.resolve(),
        policy_path=policy_paths[primary_label].resolve(),
        extra_policy_paths=tuple(path.resolve() for path in extras),
        output_tag=f"{machine_type}_{mode}_cs{setup_cost_per_machine}n",
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

