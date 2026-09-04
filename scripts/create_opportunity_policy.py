from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a compact two-threshold opportunity policy."
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--trigger", type=int, required=True)
    parser.add_argument("--opportunity_threshold", type=int, required=True)
    parser.add_argument("--env_name", default="gamma_rul_v1")
    parser.add_argument("--num_machines", type=int, required=True)
    args = parser.parse_args()
    if args.opportunity_threshold > args.trigger:
        parser.error("opportunity_threshold must not exceed trigger")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_type": "opportunity_two_threshold",
        "trigger": args.trigger,
        "opportunity_threshold": args.opportunity_threshold,
        "env_name": args.env_name,
        "num_machines": args.num_machines,
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)
    print(output_path)


if __name__ == "__main__":
    main()
