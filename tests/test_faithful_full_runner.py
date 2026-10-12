import json
from types import SimpleNamespace

from scripts.run_gamma_v3_bppo_faithful_full import (
    DEFAULT_BC_EVAL_INTERVAL,
    DEFAULT_BPPO_EVAL_INTERVAL,
    FROZEN_CLIP_RATIO,
    FROZEN_DECAY,
    completed_stage_checkpoint,
    training_is_complete,
)
from scripts.run_gamma_v3_bppo_faithful_tuning import training_command


def test_frozen_hyperparameters_match_tuning_recommendation():
    assert FROZEN_CLIP_RATIO == 0.25
    assert FROZEN_DECAY == 0.96


def test_full_suite_uses_sparse_bppo_evaluation():
    assert DEFAULT_BC_EVAL_INTERVAL == 2000
    assert DEFAULT_BPPO_EVAL_INTERVAL == 20


def test_training_command_accepts_full_suite_evaluation_intervals(tmp_path):
    args = SimpleNamespace(
        seed=456,
        num_machines=20,
        horizon=100,
        eval_episodes=10,
        eval_seed=10000,
        discount=0.95,
        device="cuda",
    )
    study = SimpleNamespace(
        dataset_path=tmp_path / "dataset.pkl",
        env_name="gamma_v3_cs100",
        policy_path=tmp_path / "policy.pkl",
        extra_policy_paths=[],
    )
    command = training_command(
        tmp_path,
        study,
        tmp_path / "run",
        args,
        bppo_steps=1000,
        clip_ratio=0.25,
        decay=0.96,
        bc_eval_interval=DEFAULT_BC_EVAL_INTERVAL,
        bppo_eval_interval=DEFAULT_BPPO_EVAL_INTERVAL,
    )

    assert command[command.index("--bc_eval_interval") + 1] == "2000"
    assert command[command.index("--bppo_eval_interval") + 1] == "20"


def test_stage_boundary_resume_requires_completion_marker(tmp_path):
    value_path = tmp_path / "value_final.pt"
    value_path.write_bytes(b"checkpoint")
    (tmp_path / "stage_status.json").write_text(
        json.dumps({"value": {"completed": False}}), encoding="utf-8")
    assert completed_stage_checkpoint(tmp_path, "value", "value_final.pt") is None

    (tmp_path / "stage_status.json").write_text(
        json.dumps({"value": {"completed": True}}), encoding="utf-8")
    assert completed_stage_checkpoint(
        tmp_path, "value", "value_final.pt") == value_path.resolve()


def test_training_completion_requires_status_and_final_checkpoint(tmp_path):
    (tmp_path / "best_checkpoint.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stage_status.json").write_text(
        json.dumps({"bppo": {"completed": True}}), encoding="utf-8")
    assert not training_is_complete(tmp_path)

    final_dir = tmp_path / "checkpoints" / "step_0001000"
    final_dir.mkdir(parents=True)
    assert training_is_complete(tmp_path)
