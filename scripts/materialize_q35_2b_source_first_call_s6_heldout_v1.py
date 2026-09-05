#!/usr/bin/env python3
"""Materialize root-selected S6 heldout coordinates without overwriting evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

SEED_SENTINEL = '"ROOT_FREEZE_REQUIRED_S6_HELDOUT_SEED"'
OFFSET_SENTINEL = '"ROOT_FREEZE_REQUIRED_S6_HELDOUT_OFFSET"'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(
    template: Path,
    output: Path,
    contract_output: Path,
    *,
    seed: int,
    instance_offset: int,
) -> dict[str, object]:
    for destination in (output, contract_output):
        if destination.exists():
            raise ValueError(f"refusing to overwrite heldout artifact: {destination}")
    if seed < 1 or instance_offset < 1:
        raise ValueError("root-selected seed and offset must be positive integers")
    source = template.read_text()
    if source.count(SEED_SENTINEL) != 2 or source.count(OFFSET_SENTINEL) != 1:
        raise ValueError("heldout template sentinels are missing or duplicated")
    rendered = source.replace(SEED_SENTINEL, str(seed)).replace(
        OFFSET_SENTINEL, str(instance_offset)
    )
    parsed = tomllib.loads(rendered)
    taskset = parsed.get("env", {}).get("taskset", {})
    sampling = parsed.get("sampling", {})
    if (
        parsed.get("num_tasks") != 16
        or parsed.get("num_rollouts") != 1
        or parsed.get("shuffle") is not False
        or taskset.get("split") != "eval"
        or taskset.get("families")
        != ["specialist_source_ast", "specialist_source_config"]
        or taskset.get("instances_per_template") != 4
        or taskset.get("seed") != seed
        or taskset.get("instance_offset") != instance_offset
        or sampling.get("seed") != seed
        or sampling.get("temperature") != 0.0
    ):
        raise ValueError("materialized S6 heldout contract failed validation")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    contract = {
        "schema_version": "q35-2b-source-first-call-s6-sampling/v1",
        "frozen_before_model_calls": True,
        "heldout_config": str(output),
        "heldout_config_sha256": digest(output),
        "seed": seed,
        "instance_offset": instance_offset,
        "template_variants": [4, 5],
        "task_count": 16,
        "family_counts": {
            "specialist_source_ast": 8,
            "specialist_source_config": 8,
        },
        "sampling": {
            key: sampling[key]
            for key in (
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "reasoning_effort",
                "max_tokens",
            )
        },
        "routing": {
            "mode": "forced_delegate_terminal_source_inspector_assignment",
            "worker_computation_scaffolded": False,
            "worker_parent_send_scaffolded": False,
        },
        "admission": {
            "minimum_treatment_hard_successes": 4,
            "minimum_hard_successes_per_family": 2,
            "minimum_paired_recoveries": 4,
            "maximum_paired_regressions": 0,
            "acceptance_gates_relaxed": False,
        },
    }
    contract_output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return {
        "heldout_config": str(output),
        "heldout_config_sha256": digest(output),
        "sampling_contract": str(contract_output),
        "sampling_contract_sha256": digest(contract_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--instance-offset", type=int, required=True)
    args = parser.parse_args()
    try:
        result = materialize(
            args.template,
            args.output,
            args.contract_output,
            seed=args.seed,
            instance_offset=args.instance_offset,
        )
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(f"S6 heldout materialization failed: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
