import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

scripts = Path(__file__).resolve().parents[1] / "scripts"
if scripts.is_dir():
    sys.path.insert(0, str(scripts))

from measure_bppo_training import BUDGETS, UpdateTimer, paths_for, trainer_arguments


class TimerTests(unittest.TestCase):
    def test_update_timing_preserves_result_and_counts_successful_call(self):
        ticks = iter([2., 5.])
        syncs = []
        progress = []
        timer = UpdateTimer(lambda: syncs.append(True), lambda: progress.append(True), lambda: next(ticks))

        class Learner:
            def update(self, x, y=1):
                return x + y

        original = Learner.update
        try:
            timer.wrap_update(Learner, "value")
            self.assertEqual(Learner().update(4, y=3), 7)
            self.assertEqual(timer.counts["value"], 1)
            self.assertEqual(timer.seconds["value"], 3.)
            self.assertEqual(len(syncs), 2)
            self.assertEqual(progress, [])
        finally:
            timer.restore()
        self.assertIs(Learner.update, original)

    def test_inherited_update_is_restored_without_changing_base(self):
        class Base:
            def update(self):
                return 9

        class Child(Base):
            pass

        ticks = iter([0., 1.])
        timer = UpdateTimer(lambda: None, lambda: None, lambda: next(ticks))
        original = Base.update
        timer.wrap_update(Child, "q")
        self.assertEqual(Child().update(), 9)
        timer.restore()
        self.assertNotIn("update", Child.__dict__)
        self.assertIs(Base.update, original)

    def test_evaluation_time_is_separate(self):
        ticks = iter([1., 6.])
        module = SimpleNamespace(evaluate_policy_fn=lambda value: value * 2)
        original = module.evaluate_policy_fn
        timer = UpdateTimer(lambda: None, lambda: None, lambda: next(ticks))
        timer.wrap_evaluation(module)
        self.assertEqual(module.evaluate_policy_fn(3), 6)
        self.assertEqual(timer.evaluation_seconds, 5.)
        self.assertEqual(timer.snapshot()["optimization_seconds"], 0.)
        self.assertFalse(timer.snapshot()["all_stages_complete"])
        timer.restore()
        self.assertIs(module.evaluate_policy_fn, original)

    def test_full_budgets_are_required_for_completion(self):
        timer = UpdateTimer(lambda: None, lambda: None)
        timer.counts = dict(BUDGETS)
        self.assertTrue(timer.snapshot()["all_stages_complete"])
        timer.counts["value"] = 0
        self.assertFalse(timer.snapshot()["all_stages_complete"])

    def test_command_has_frozen_budgets_and_no_checkpoint_reuse(self):
        args = argparse.Namespace(dt_repo=Path("dt"), output_root=Path("timing"), seed=123, device="cuda")
        env, dataset, output = paths_for(args, 40)
        command = trainer_arguments(args, 40, dataset, output)
        self.assertEqual(env, "gamma_v3_cs1000")
        for option, expected in (("--v_steps", "2000000"), ("--q_bc_steps", "2000000"),
                                 ("--bc_steps", "500000"), ("--bppo_steps", "1000"),
                                 ("--bppo_eval_interval", "20")):
            self.assertEqual(command[command.index(option) + 1], expected)
        self.assertFalse(any(x in command for x in ("--value_checkpoint", "--q_checkpoint", "--bc_checkpoint")))


if __name__ == "__main__":
    unittest.main()
