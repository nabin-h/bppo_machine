import importlib.util
from pathlib import Path
import tempfile
import unittest


script = Path(__file__).with_name("estimate_bppo_evaluation_overhead.py")
if not script.is_file():
    script = Path(__file__).resolve().parents[1] / "scripts" / "estimate_bppo_evaluation_overhead.py"
spec = importlib.util.spec_from_file_location("overhead", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvaluationOverheadTests(unittest.TestCase):
    def config(self, interval):
        return dict(bc_steps=500000, bc_eval_interval=2000,
                    bppo_steps=1000, bppo_eval_interval=interval)

    def history(self, interval):
        return ([dict(stage="bc") for _ in range(250)]
                + [dict(stage="bppo_initial_bc")]
                + [dict(stage="bppo") for _ in range(1000 // interval)])

    def test_original_dense_schedule(self):
        self.assertEqual(module.evaluation_counts(self.config(1), self.history(1)),
                         dict(bc=250, bppo_initial_bc=1, bppo=1000))

    def test_sparse_schedule(self):
        self.assertEqual(module.evaluation_counts(self.config(20), self.history(20)),
                         dict(bc=250, bppo_initial_bc=1, bppo=50))

    def test_final_step_counted_once_when_not_divisible(self):
        config = dict(bc_steps=5, bc_eval_interval=2, bppo_steps=5, bppo_eval_interval=2)
        history = ([dict(stage="bc")] * 3 + [dict(stage="bppo_initial_bc")]
                   + [dict(stage="bppo")] * 3)
        self.assertEqual(module.evaluation_counts(config, history)["bppo"], 3)

    def test_incomplete_history_rejected(self):
        with self.assertRaises(ValueError):
            module.evaluation_counts(self.config(20), self.history(20)[:-1])

    def test_subtraction_and_sensitivity(self):
        result = module.adjusted_estimate(100, dict(bc=2, bppo=1),
                                          dict(bc=[2, 4], bppo=[5, 7]))
        self.assertEqual(result["estimated_evaluation_seconds"], 12)
        self.assertEqual(result["estimated_evaluation_excluded_seconds"], 88)
        self.assertEqual(result["sensitivity_seconds"], [85, 91])

    def test_impossible_subtraction_rejected(self):
        with self.assertRaises(ValueError):
            module.adjusted_estimate(10, dict(bc=10), dict(bc=[1, 2]))

    def test_foreign_windows_username_relocated(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            repo = Path(tmp) / "bppo_machine"
            target = repo / "policies" / "reference.pkl"
            target.parent.mkdir(parents=True)
            target.touch()
            self.assertEqual(module.relocated_path(
                r"C:\Users\other\myPapers\BPPO_machine\policies\reference.pkl", repo), target)

    def test_summary_is_partial_until_all_eligible_runs_measured(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            report = dict(num_machines=20, estimated_evaluation_excluded_seconds=3600)
            result = module.write_summary(Path(tmp), [report], [], expected=2)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["rows"][0]["mean_hours"], 1)


if __name__ == "__main__":
    unittest.main()
