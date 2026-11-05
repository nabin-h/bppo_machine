# BPPO evaluation-overhead estimate

Run `estimate_bppo_evaluation_overhead.py` in the original BPPO Python environment
on the original training hardware. No training is performed. Do not run another
GPU job concurrently. Original result files are read only.

Required inputs: the full original run folders, including `training_config.json`,
`timing.json`, `<seed>_eval.json`, `bc_best.pt`, and BPPO checkpoints. The compact
October result archive may omit checkpoints; use the original training directory.
Original reference-policy pickle files must also be present. Their matching
diagnostics are retained in every timed evaluation.

Example from the BPPO repository:

```bat
python scripts\estimate_bppo_evaluation_overhead.py --bppo-repo . --runs-root "Seed runs\gamma_v3_bppo_faithful_full" --output-root "Seed runs\bppo_evaluation_overhead" --dry-run
python scripts\estimate_bppo_evaluation_overhead.py --bppo-repo . --runs-root "Seed runs\gamma_v3_bppo_faithful_full" --output-root "Seed runs\bppo_evaluation_overhead" --repeats 3 --warmups 1 --confirm-same-hardware
```

The dry-run does not need PyTorch and lists missing artifacts and exclusions.
Measurement requires CUDA and confirmation that hardware matches the original.
Each saved policy receives one untimed full-evaluation warmup and three timed
evaluations by default. BC and initial BPPO are measured using `bc_best.pt`; later
BPPO evaluations use available best/early/final snapshots. Each evaluation uses
the original episode count, horizon, discount, seed, feasibility rules, and
reference policies. CUDA is synchronized around measured calls. No datasets or
value/Q networks are loaded, and no gradient updates are performed.

Counts are verified against the saved evaluation history and configured schedule:
500k BC steps at interval 2000 give 250 calls (steps 1, 2001, ...);
BPPO adds one initial evaluation plus 1000 calls at interval 1, or 50 at interval
20. Baseline evaluations precede the original training timer. External final
200-episode reevaluations occur after it. Neither is subtracted.

Estimated evaluation overhead is the sum of stage-specific mean measured call
times multiplied by historical call counts. The residual includes optimization,
logging, checkpoint saving, and other training-loop overhead; it is not pure GPU
optimization time. Early BC checkpoints were not saved, so late BC timing is a
proxy. Hardware/load and policy-dependent timing differences introduce uncertainty.
Propagated observed minimum/maximum call times are descriptive sensitivity ranges,
not statistical confidence intervals. Full-budget runs that reused pretraining are
excluded. Measurements that could subtract more than the recorded time fail.

Outputs:
- `summary.json`: final or partial status, exclusions, counts and table aggregates.
- `table.md`: estimated hours, mean (minimum-maximum), grouped by system size.
- `table.tex`: the same table in LaTeX format.
- Per-run `evaluation_overhead.json`: raw samples, checkpoint/source hashes,
  hardware, call counts, original time, estimated overhead and adjusted residual.

Interrupted measurements can be rerun. Matching completed run measurements are
reused only if the input hashes, code, hardware and measurement settings match.
Do not report the table as complete unless `summary.json` says `complete` and the
expected eligible run coverage is present. This procedure does not infer historical
GPU memory usage or replace a controlled equal-budget compute benchmark.
