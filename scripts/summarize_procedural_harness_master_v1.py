"""Summarize procedural Harness Master traces by split, family, and hard gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _traces(path: Path) -> list[dict[str, Any]]:
    trace_file = path / "traces.jsonl" if path.is_dir() else path
    if not trace_file.is_file():
        raise ValueError(f"missing traces: {trace_file}")
    return [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]


def _score(trace: dict[str, Any], name: str, collection: str) -> float:
    value = trace.get(collection, {}).get(name, 0.0)
    if isinstance(value, dict):
        value = value.get("score", value.get("value", 0.0))
    return float(value) if isinstance(value, (int, float)) else 0.0


def summarize(paths: list[Path]) -> dict[str, Any]:
    rows = [trace for path in paths for trace in _traces(path)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostics = defaultdict(float)
    for trace in rows:
        data = trace.get("task", {}).get("data", {})
        grouped[str(data.get("family", "unknown"))].append(trace)
        for key in (
            "final_answer_exact",
            "all_required_atoms",
            "no_forbidden_atoms",
            "ordering_satisfied",
            "cardinality_exact",
            "required_atoms_fraction",
        ):
            diagnostics[key] += _score(trace, key, "metrics")

    def cohort(values: list[dict[str, Any]]) -> dict[str, Any]:
        passed = sum(_score(trace, "harness_score", "rewards") == 1.0 for trace in values)
        return {"episodes": len(values), "passed": passed, "rate": passed / len(values) if values else 0.0}

    return {
        "episodes": len(rows),
        "harness": cohort(rows),
        "by_family": {family: cohort(values) for family, values in sorted(grouped.items())},
        "diagnostic_means": {
            key: value / len(rows) if rows else 0.0 for key, value in sorted(diagnostics.items())
        },
        "errors": sum(bool(trace.get("error") or trace.get("errors")) for trace in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(summarize(args.paths), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
