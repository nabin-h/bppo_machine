# Representative BPPO optimization timing

This wrapper runs the existing gamma-v3 trainer without editing its source.
It instruments update calls for value, Q, behavior cloning, and BPPO. CUDA is
synchronized at both update timing boundaries. Evaluation is timed separately.
Checkpointing, logging, initialization, and dataset loading are outside the
optimization timer. Minibatch sampling inside each update is included.

## Fixed experiment

- Homogeneous M20/Cs500 and M40/Cs1000, plausible-policy data.
- Seed 123, 20,000 existing trajectories, 100 steps, discount 0.95.
- All stages trained from scratch: 2m value, 2m Q, 500k BC, 1000 BPPO.
- Original network sizes and frozen hyperparameters.
- BC evaluation every 2000 updates and BPPO evaluation every 20 updates,
  ten episodes each, evaluation seed 10000.
- Feasibility handling and policy selection remain in the original trainer.
- Optional reference-policy matching diagnostics are omitted; they do not
  determine the selected cost or optimizer update. This is a timing experiment,
  not a replacement for the final performance comparisons.
- Each size runs in its own process, sequentially. No GPU job is launched by
  a dry run. Do not run alongside other training jobs.

Use the activated BPPO PyTorch/CUDA environment on the experiment computer:

```bat
python scripts\measure_bppo_training.py --components 20,40 --seed 123 --device cuda --dry-run
python scripts\measure_bppo_training.py --components 20,40 --seed 123 --device cuda
```

Run these from the BPPO repository directory. The installed script belongs in
`scripts/measure_bppo_training.py`, with its test in
`tests/test_measure_bppo_training.py`. It finds the adjacent `dt_machine` repo
by default; override with `--dt-repo` if needed. Results default to
`Seed runs/bppo_optimization_timing/`. `optimization_timing.json` records per-stage update counts,
synchronized optimization hours, evaluation seconds, source hashes, and actual
hardware. `representative_timing_summary.json` consolidates completed runs.
The original trainer's `timing.json` is still its inclusive wall-clock measure;
do not mistake it for the wrapper's optimization-only measurement.

The wrapper saves partial counters every 10,000 updates. A completed run is
skipped on rerun. Partial runs cannot be reused for a full from-scratch timing:
select a new output directory with `--output-root` instead of overwriting them.
Report these as representative single-seed measurements, not across-seed means.
No full timing has been measured on the preparation computer, which lacks the
required PyTorch/CUDA runtime and benchmark datasets.

The preparatory copy under `iql_machine/for v3/bppo_timing/` also remains usable,
but the commands above refer to the version distributed in the BPPO repository.
