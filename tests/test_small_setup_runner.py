import sys

from scripts import gamma_v3_experiment as experiment
from scripts import run_gamma_v3_bppo_small_setup_batch as small_batch


def test_small_environment_matrix_is_complete_and_unique():
    assert len(small_batch.SMALL_ENVIRONMENTS) == 8
    assert {key[2] for key in small_batch.SMALL_ENVIRONMENTS} == {5, 10}
    assert {key[0] for key in small_batch.SMALL_ENVIRONMENTS} == {
        "homogeneous",
        "heterogeneous",
    }
    assert {key[1] for key in small_batch.SMALL_ENVIRONMENTS} == {5, 25}
    assert len(set(small_batch.SMALL_ENVIRONMENTS.values())) == 8


def test_small_batch_injects_defaults_and_preserves_overrides(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        small_batch.batch,
        "main",
        lambda: captured.update(arguments=sys.argv[1:]),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner.py", "--seeds", "7", "--prepare_only"],
    )

    small_batch.main()

    arguments = captured["arguments"]
    assert arguments[arguments.index("--num_machines") + 1] == "5,10"
    assert arguments[arguments.index("--seeds") + 1] == "7"
    assert arguments[arguments.index("--output_root") + 1].endswith(
        "gamma_v3_bppo_faithful_small"
    )
    assert "--resume" in arguments
    for key, env_name in small_batch.SMALL_ENVIRONMENTS.items():
        assert experiment.ENVIRONMENTS[key] == env_name
