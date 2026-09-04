from __future__ import annotations

import argparse
from pathlib import Path

from prime_rl.latent.audit import load_and_validate_launch_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail closed unless the A0/A1 plan is launch-ready.")
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    load_and_validate_launch_plan(args.plan)


if __name__ == "__main__":
    main()
