from __future__ import annotations

import argparse
import json
from pathlib import Path

from prime_rl.latent.transfer_bank import build_transfer_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic split-information bank.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("train", "validation", "held_out"), required=True)
    parser.add_argument("--examples-per-family", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bank = build_transfer_bank(
        seed=args.seed,
        split=args.split,
        examples_per_family=args.examples_per_family,
    )
    args.output.write_text(json.dumps(bank.artifact_dict(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
