# Faithful BPPO gamma-v3 workflow

The gamma-v3 adaptation now matches the original BPPO training recipe wherever the maintenance problem permits:

- value: 2,000,000 updates, 512-wide, depth 3;
- Q: 2,000,000 updates, 1024-wide, depth 2;
- behavior cloning: 500,000 updates, 1024-wide, depth 2;
- BPPO: 1,000 updates, 1024-wide, depth 2;
- learning rate `1e-4`, batch size 512, target update frequency 2, `tau=0.005`, `omega=0.9`, and zero entropy weight;
- clip and BPPO learning-rate decay for the first 200 updates;
- evaluation after every BPPO update and replacement of the old policy whenever evaluation improves.

The intentional adaptations are a factorized Bernoulli policy for the binary multi-action space, linear value/Q outputs for negative-cost rewards, and benchmark discount `gamma=0.95`. Environment action projection remains outside the policy, as before, but evaluation now reports forced-failure, suppressed-zero-state, and other correction counts and rates separately.

## Tune clip ratio and decay first

The tuning runner validates the exact DT dataset and metadata, trains value/Q/BC once, then reuses those checkpoints for the six paper-supported combinations of clip ratio `{0.1, 0.25}` and decay `{0.94, 0.96, 0.98}`. It evaluates both the author-style selected checkpoint and the final BPPO checkpoint over 200 held-out episodes.

```text
python scripts\run_gamma_v3_bppo_faithful_tuning.py --device cuda --resume
```

The representative default task is heterogeneous, `Cs=25N`, `M=20`, plausible-policy data, seed 123. Results are written below `Seed runs/gamma_v3_bppo_faithful_tuning`. The shared stages are resumable at stage boundaries: a completed value, Q, or BC checkpoint is reused after interruption.

Use `--prepare_only` to validate inputs and write the complete tuning specification without training. Use `--clip_ratios` or `--decays` to run a subset.

## Full comparison after freezing hyperparameters

The full runner also defaults to the faithful budgets, architectures, decay controls, and per-update BPPO evaluation:

```text
python run_bppo_gamma_v3_comparison.py --machine_types heterogeneous --setup_cost_per_machine 25 --num_machines 20 --modes plausible --seeds 123 --device cuda --resume
```

It uses the exact datasets in the sibling `dt_machine` repository and validates environment name, system size, horizon, discount, dataset seed, setup cost, and reward convention. The training-selected checkpoint is independently evaluated for 200 episodes with seed 20000. Results are written below `Seed runs/gamma_v3_comparison`.

To continue a suite when some DT datasets are absent, add `--skip_missing_datasets --resume`. Missing inputs are recorded in `Seed runs/gamma_v3_comparison/bppo_missing_datasets.json`.
