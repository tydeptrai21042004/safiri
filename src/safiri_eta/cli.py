from __future__ import annotations

import argparse
import json
from typing import Any

from .config import load_config
from .evaluation import evaluate
from .models import train_models
from .reporting import generate_technical_report
from .simulation import generate_dataset
from .snapshots import build_snapshots
from .validation import normalize_events


def run_all(config: dict[str, Any], n_shipments: int | None = None) -> dict[str, Any]:
    outputs = {
        "generation": {key: str(value) for key, value in generate_dataset(config, n_shipments).items()},
        "validation": {key: str(value) for key, value in normalize_events(config).items()},
        "snapshots": {key: str(value) for key, value in build_snapshots(config).items()},
        "model": str(train_models(config)),
    }
    outputs["metrics"] = evaluate(config)
    outputs["report"] = str(generate_technical_report(config))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="SAFiRi milestone-aware ETA pipeline")
    parser.add_argument("command", choices=["generate", "prepare", "train", "evaluate", "report", "all"])
    parser.add_argument("--config", default=None, help="Path to YAML configuration")
    parser.add_argument("--n-shipments", type=int, default=None, help="Override synthetic shipment count")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "generate":
        result = generate_dataset(config, args.n_shipments)
    elif args.command == "prepare":
        result = {"validation": normalize_events(config), "snapshots": build_snapshots(config)}
    elif args.command == "train":
        result = {"model": train_models(config)}
    elif args.command == "evaluate":
        result = evaluate(config)
    elif args.command == "report":
        result = {"report": generate_technical_report(config)}
    else:
        result = run_all(config, args.n_shipments)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

