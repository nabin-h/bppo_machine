# BPPO gamma-v3 comparison

This workflow is additive. Existing BPPO environments, scripts, configurations, and results are unchanged.

The runner uses the exact datasets stored in the sibling `dt_machine` repository and validates the environment name, system size, horizon, discount, dataset seed, setup cost, and reward convention before training. It supports homogeneous and heterogeneous machines, setup costs `5N` and `25N`, `M=20` and `M=40`, and plausible-policy and threshold-only datasets.

Prepare and validate one study without training:

```text
python run_bppo_gamma_v3_comparison.py --machine_types heterogeneous --setup_cost_per_machine 25 --num_machines 20 --modes plausible --prepare_only
```

Run the same study:

```text
python run_bppo_gamma_v3_comparison.py --machine_types heterogeneous --setup_cost_per_machine 25 --num_machines 20 --modes plausible --seeds 123 --device cuda
```

The existing BPPO architecture and optimization settings are retained. By default, value and Q/BC training use 20,000 steps, BC uses 10,000 steps, and BPPO uses 5,000 steps. The training-selected checkpoint is independently evaluated for 200 episodes with seed 20000. Results are written under `Seed runs/gamma_v3_comparison`.

