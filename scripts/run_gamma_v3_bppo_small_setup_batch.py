from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import gamma_v3_experiment as experiment  # noqa: E402
from scripts import run_gamma_v3_bppo_faithful_full as batch  # noqa: E402


SMALL_ENVIRONMENTS = {
    ("homogeneous", 5, 5): "gamma_v3_cs25",
    ("homogeneous", 25, 5): "gamma_v3_cs125",
    ("homogeneous", 5, 10): "gamma_v3_cs50",
    ("homogeneous", 25, 10): "gamma_v3_cs250",
    ("heterogeneous", 5, 5): "gamma_v3_hetero_cs5n_m5",
    ("heterogeneous", 25, 5): "gamma_v3_hetero_cs25n_m5",
    ("heterogeneous", 5, 10): "gamma_v3_hetero_cs5n_m10",
    ("heterogeneous", 25, 10): "gamma_v3_hetero_cs25n_m10",
}


def has_argument(arguments: list[str], name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in arguments)


def main() -> None:
    # The validated full-suite runner imports helpers from this module. Extending
    # the shared mapping here keeps the M20/M40 runner unchanged while allowing
    # the same dataset validation and policy construction for M5/M10.
    experiment.ENVIRONMENTS.update(SMALL_ENVIRONMENTS)

    arguments = sys.argv[1:]
    defaults = (
        ("--num_machines", "5,10"),
        ("--seeds", "123,456,789"),
        (
            "--output_root",
            str(REPO_ROOT / "Seed runs" / "gamma_v3_bppo_faithful_small"),
        ),
    )
    for name, value in defaults:
        if not has_argument(arguments, name):
            arguments.extend((name, value))
    if "--resume" not in arguments:
        arguments.append("--resume")

    sys.argv = [sys.argv[0], *arguments]
    batch.main()


if __name__ == "__main__":
    main()
