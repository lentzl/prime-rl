"""Summarize procedural Harness Master traces by split, family, and hard gate."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _traces(path: Path) -> list[dict[str, Any]]:
    trace_file = path / "traces.jsonl" if path.is_dir() else path
    if not trace_file.is_file():
        raise ValueError(f"missing traces: {trace_file}")
    traces = []
    for line in trace_file.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        nested = record.get("traces")
        traces.extend(nested if isinstance(nested, list) else [record])
    return traces


def _score(trace: dict[str, Any], name: str, collection: str) -> float:
    value = trace.get(collection, {}).get(name, 0.0)
    if isinstance(value, dict):
        value = value.get("score", value.get("value", 0.0))
    return float(value) if isinstance(value, (int, float)) else 0.0


def _rescore(trace: dict[str, Any]) -> dict[str, Any]:
    import verifiers.v1 as vf
    from procedural_harness_master_v1.taskset import (
        ProceduralHarnessMasterData,
        _contract_behavior,
    )

    rescored = copy.deepcopy(trace)
    data = ProceduralHarnessMasterData.model_validate(trace["task"]["data"])
    behavior = _contract_behavior(vf.Trace.model_validate(trace), data)
    rescored.setdefault("metrics", {}).update(behavior)
    reward = rescored.setdefault("rewards", {}).setdefault("harness_score", {})
    if isinstance(reward, dict):
        reward["score"] = behavior["harness_score"]
    else:
        rescored["rewards"]["harness_score"] = behavior["harness_score"]
    return rescored


def summarize(paths: list[Path], *, rescore: bool = False) -> dict[str, Any]:
    rows = [trace for path in paths for trace in _traces(path)]
    if rescore:
        rows = [_rescore(trace) for trace in rows]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_families: dict[str, str] = {}
    diagnostics = defaultdict(float)
    for trace in rows:
        data = trace.get("task", {}).get("data", {})
        family = str(data.get("family", "unknown"))
        task_key = str(
            data.get("episode_id")
            or trace.get("task", {}).get("key")
            or trace.get("task", {}).get("hash")
            or trace.get("id")
        )
        grouped[family].append(trace)
        task_groups[task_key].append(trace)
        task_families[task_key] = family
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

    def comparison_groups(groups: list[list[dict[str, Any]]]) -> dict[str, int]:
        pass_counts = [cohort(values)["passed"] for values in groups]
        return {
            "groups": len(groups),
            "informative": sum(
                0 < passed < len(values)
                for passed, values in zip(pass_counts, groups)
            ),
            "all_pass": sum(
                passed == len(values)
                for passed, values in zip(pass_counts, groups)
            ),
            "all_fail": sum(passed == 0 for passed in pass_counts),
        }

    family_task_groups: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for task_key, values in task_groups.items():
        family_task_groups[task_families[task_key]].append(values)

    return {
        "rescored": rescore,
        "episodes": len(rows),
        "harness": cohort(rows),
        "by_family": {family: cohort(values) for family, values in sorted(grouped.items())},
        "comparison_groups": comparison_groups(list(task_groups.values())),
        "by_family_groups": {
            family: comparison_groups(values)
            for family, values in sorted(family_task_groups.items())
        },
        "diagnostic_means": {
            key: value / len(rows) if rows else 0.0 for key, value in sorted(diagnostics.items())
        },
        "errors": sum(bool(trace.get("error") or trace.get("errors")) for trace in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(summarize(args.paths, rescore=args.rescore), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
