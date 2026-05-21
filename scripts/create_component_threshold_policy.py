import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Create a compact component-threshold policy pickle for reference evaluation and dataset generation."
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--env_name", type=str, default="env3")
    parser.add_argument("--num_machines", type=int, required=True)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "policy_type": "component_threshold",
        "threshold": int(args.threshold),
        "env_name": str(args.env_name),
        "num_machines": int(args.num_machines),
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(output_path)


if __name__ == "__main__":
    main()
