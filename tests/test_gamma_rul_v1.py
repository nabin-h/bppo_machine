from __future__ import annotations

import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch.nn as nn

try:
    import gym  # noqa: F401
except ModuleNotFoundError:
    gym_stub = types.ModuleType("gym")
    gym_stub.Env = object
    sys.modules["gym"] = gym_stub

from custom_env import (
    cost_function,
    get_env_spec,
    opportunity_threshold_policy,
    sample_initial_observation,
    take_step,
)
from evaluation import reference_action_for_state
from maintenance_dataset import load_trajectory_dataset
from net import QMLP, ValueMLP


ENV_NAME = "gamma_rul_v1"


class GammaRulEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = get_env_spec(ENV_NAME)
        self.transition = np.asarray(
            self.spec["transition_no_maintenance"], dtype=np.float64
        )

    def test_transition_and_expected_lifetime(self) -> None:
        np.testing.assert_allclose(self.transition.sum(axis=1), 1.0, atol=1e-12)
        transient = self.transition[:-1, :-1]
        expected_life = np.linalg.solve(
            np.eye(9) - transient, np.ones(9, dtype=np.float64)
        )
        self.assertAlmostEqual(float(expected_life[0]), 100.0, places=8)
        np.testing.assert_allclose(
            expected_life,
            np.asarray(self.spec["expected_remaining_life"])[:-1],
            atol=1e-8,
        )

    def test_cost_and_opportunity_policy(self) -> None:
        cost = cost_function([6, 8, 9], [1, 1, 1], env_name=ENV_NAME)
        self.assertAlmostEqual(cost, 369.9062668581847, places=8)
        action = opportunity_threshold_policy(
            [5, 6, 7, 8, 9], env_name=ENV_NAME
        )
        np.testing.assert_array_equal(action, [0, 1, 1, 1, 1])

    def test_maintenance_resets_components(self) -> None:
        next_state, _ = take_step(
            [6, 8, 9],
            [1, 1, 1],
            rng=np.random.RandomState(12),
            env_name=ENV_NAME,
        )
        np.testing.assert_array_equal(next_state, [0, 0, 0])

    def test_burnin_initialization_is_reproducible(self) -> None:
        first = sample_initial_observation(
            50, np.random.RandomState(123), env_name=ENV_NAME
        )
        second = sample_initial_observation(
            50, np.random.RandomState(123), env_name=ENV_NAME
        )
        np.testing.assert_array_equal(first, second)

    def test_opportunity_reference_action(self) -> None:
        policy = {
            "policy_type": "opportunity_two_threshold",
            "trigger": 8,
            "opportunity_threshold": 6,
        }
        action = reference_action_for_state(policy, (5, 6, 7, 8, 9), 5)
        np.testing.assert_array_equal(action, [0, 1, 1, 1, 1])


class BppoObjectiveTests(unittest.TestCase):
    def test_loader_preserves_negative_cost_reward(self) -> None:
        trajectories = [
            {
                "states": [[1], [2]],
                "actions": [[0], [1]],
                "rewards": [-3.0, -7.0],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.pkl"
            with dataset_path.open("wb") as handle:
                pickle.dump(trajectories, handle)
            dataset = load_trajectory_dataset(str(dataset_path))
        np.testing.assert_array_equal(dataset["rewards"].reshape(-1), [-3.0, -7.0])

    def test_critics_can_represent_negative_returns(self) -> None:
        self.assertIsInstance(ValueMLP(2, 8, 2)._net[-1], nn.Linear)
        self.assertIsInstance(QMLP(2, 2, 8, 2)._net[-1], nn.Linear)


if __name__ == "__main__":
    unittest.main()
