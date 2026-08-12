#!/usr/bin/env python3
"""Compare paired natural follow-up and handshake candidate screens."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    traces: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for trace in record.get("traces") or [record]:
            task = trace["task"]["data"]
            key = (task["family"], task["template_variant"], task["idx"])
            if key in traces:
                raise ValueError(f"duplicate task key in {path}: {key}")
            traces[key] = trace
    return traces


def _metric(trace: dict[str, Any], name: str) -> float:
    metrics = trace.get("metrics") or {}
    if name not in metrics:
        raise ValueError(f"trace {trace.get('id')} has no {name!r} metric")
    return float(metrics[name])


def _summaries(path: Path) -> dict[str, dict[str, float]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, trace in _load(path).items():
        by_family[key[0]].append(trace)
    result: dict[str, dict[str, float]] = {}
    for family, traces in sorted(by_family.items()):
        result[family] = {
            "tasks": len(traces),
            "answer_accuracy": sum(_metric(trace, "answer_accuracy") for trace in traces),
            "natural_followup_causal": sum(_metric(trace, "natural_followup_causal") for trace in traces),
            "protocol_aligned": sum(_metric(trace, "protocol_aligned") for trace in traces),
            "clean_protocol_aligned": sum(_metric(trace, "clean_protocol_aligned") for trace in traces),
            "coordinator_delegated_path_accesses": sum(
                _metric(trace, "coordinator_delegated_path_accesses") for trace in traces
            ),
            "bidirectional_control_mean": sum(_metric(trace, "bidirectional_control") for trace in traces)
            / len(traces),
        }
    return result


def summarize(base_path: Path, candidate_path: Path) -> dict[str, Any]:
    base = _summaries(base_path)
    candidate = _summaries(candidate_path)
    if base.keys() != candidate.keys():
        raise ValueError("base and candidate family sets differ")
    family_passes: dict[str, bool] = {}
    for family in base:
        family_passes[family] = (
            candidate[family]["answer_accuracy"] >= base[family]["answer_accuracy"]
            and candidate[family]["natural_followup_causal"] >= base[family]["natural_followup_causal"]
            and candidate[family]["protocol_aligned"] >= base[family]["protocol_aligned"]
            and candidate[family]["bidirectional_control_mean"] >= base[family]["bidirectional_control_mean"]
        )
    zero_candidate_path_access = all(
        values["coordinator_delegated_path_accesses"] == 0 for values in candidate.values()
    )
    return {
        "base": base,
        "candidate": candidate,
        "family_passes": family_passes,
        "zero_candidate_coordinator_path_access": zero_candidate_path_access,
        "promotion_pass": all(family_passes.values()) and zero_candidate_path_access,
        "criteria": {
            "per_family": [
                "answer_accuracy does not decrease",
                "natural_followup_causal does not decrease",
                "protocol_aligned does not decrease",
                "mean bidirectional_control does not decrease",
            ],
            "candidate_coordinator_delegated_path_accesses": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.base, args.candidate)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
