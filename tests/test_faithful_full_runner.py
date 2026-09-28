import json

from scripts.run_gamma_v3_bppo_faithful_full import (
    FROZEN_CLIP_RATIO,
    FROZEN_DECAY,
    completed_stage_checkpoint,
    training_is_complete,
)


def test_frozen_hyperparameters_match_tuning_recommendation():
    assert FROZEN_CLIP_RATIO == 0.25
    assert FROZEN_DECAY == 0.96


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
