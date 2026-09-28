import numpy as np

from evaluation import classify_action_corrections


def test_classify_action_corrections_separates_projection_causes():
    counts = classify_action_corrections(
        observation=np.asarray([3, 0, 2], dtype=np.int32),
        raw_action=np.asarray([0, 1, 0], dtype=np.int32),
        projected_action=np.asarray([1, 0, 1], dtype=np.int32),
    )

    assert counts == {
        "corrections": 3,
        "forced_failure_corrections": 1,
        "suppressed_zero_state_corrections": 1,
        "other_corrections": 1,
    }
