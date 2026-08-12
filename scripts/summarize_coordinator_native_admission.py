#!/usr/bin/env python3
"""Summarize strict native success supply for coordinator-owned tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MINIMUM_SUCCESS_FAMILIES = 6
MINIMUM_MULTI_SUCCESS_FAMILIES = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_traces(path: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        payload = json.loads(line)
        nested = payload.get("traces")
        if not isinstance(nested, list) or len(nested) != 1:
            raise ValueError(f"trace line {line_number} must contain exactly one nested trace")
        traces.append(nested[0])
    return traces


def summarize(path: Path) -> dict[str, Any]:
    traces = _iter_traces(path)
    successes: Counter[str] = Counter()
    rollouts: Counter[str] = Counter()
    components: Counter[str] = Counter()
    trace_ids: defaultdict[str, list[str]] = defaultdict(list)
    for trace in traces:
        data = trace["task"]["data"]
        family = data["resource_family"]
        metrics = trace["metrics"]
        rollouts[family] += 1
        for key, value in metrics.items():
            components[key] += int(float(value) == 1.0)
        if float(metrics["strict_success"]) == 1.0:
            successes[family] += 1
            trace_ids[family].append(trace["id"])

    success_families = sorted(successes)
    multi_success_families = sorted(family for family, count in successes.items() if count >= 2)
    admission_pass = (
        len(success_families) >= MINIMUM_SUCCESS_FAMILIES
        and len(multi_success_families) >= MINIMUM_MULTI_SUCCESS_FAMILIES
    )
    return {
        "traces": len(traces),
        "trace_sha256": _sha256(path),
        "strict_successes": sum(successes.values()),
        "success_families": success_families,
        "multi_success_families": multi_success_families,
        "admission_pass": admission_pass,
        "criteria": {
            "minimum_success_families": MINIMUM_SUCCESS_FAMILIES,
            "minimum_multi_success_families": MINIMUM_MULTI_SUCCESS_FAMILIES,
        },
        "components": dict(sorted(components.items())),
        "families": {
            family: {
                "rollouts": rollouts[family],
                "strict_successes": successes[family],
                "success_trace_ids": trace_ids[family],
            }
            for family in sorted(rollouts)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(summarize(args.traces), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
