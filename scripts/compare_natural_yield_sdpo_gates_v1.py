"""Compare R7 and a natural-yield SDPO candidate on identical frozen gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.summarize_procedural_harness_master_v1 import _traces

GATES = {
    "natural_yield": "natural-yield",
    "natural_yield_local_work": "natural-yield-local-work",
    "atomic_state": "atomic-state",
    "atomic_send": "atomic-send",
}

LOCAL_WORK_DIAGNOSTICS = {
    "local_work_before_yield",
    "premature_yield_before_local_work",
}


class ComparisonFailure(ValueError):
    """The gate evidence cannot support a same-draw comparison."""


def _task_signature(trace: dict[str, Any]) -> str:
    data = trace.get("task", {}).get("data", {})
    stable = {
        key: data.get(key)
        for key in ("episode_id", "family", "prompt", "system_prompt", "oracle")
    }
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def _read_gate(root: Path, label: str, suffix: str) -> dict[str, Any]:
    run = root / f"{label}-{suffix}" / "train-admission"
    summary_path = run / "SUMMARY.json"
    if not summary_path.is_file():
        raise ComparisonFailure(f"missing gate summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("rescored") is not True or summary.get("episodes") != 8:
        raise ComparisonFailure(f"gate is not an eight-episode rescored run: {run}")
    if summary.get("errors") != 0:
        raise ComparisonFailure(f"gate contains rollout errors: {run}")
    traces = _traces(run)
    if len(traces) != 8:
        raise ComparisonFailure(f"gate contains {len(traces)} traces instead of eight")
    signatures = {_task_signature(trace) for trace in traces}
    if len(signatures) != 1:
        raise ComparisonFailure(f"gate contains multiple task specifications: {run}")
    harness = summary.get("harness", {})
    return {
        "task_signature": signatures.pop(),
        "passed": harness.get("passed"),
        "rate": harness.get("rate"),
        "diagnostic_means": summary.get("diagnostic_means", {}),
    }


def compare(root: Path, base_label: str, candidate_label: str) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for name, suffix in GATES.items():
        base = _read_gate(root, base_label, suffix)
        candidate = _read_gate(root, candidate_label, suffix)
        if base["task_signature"] != candidate["task_signature"]:
            raise ComparisonFailure(f"{name} task specifications differ")
        diagnostic_keys = sorted(
            set(base["diagnostic_means"]) | set(candidate["diagnostic_means"])
        )
        gates[name] = {
            "base": {key: value for key, value in base.items() if key != "task_signature"},
            "candidate": {
                key: value for key, value in candidate.items() if key != "task_signature"
            },
            "delta_passed": candidate["passed"] - base["passed"],
            "diagnostic_deltas": {
                key: candidate["diagnostic_means"].get(key, 0.0)
                - base["diagnostic_means"].get(key, 0.0)
                for key in diagnostic_keys
            },
        }

    target_improved = gates["natural_yield"]["delta_passed"] > 0
    prerequisites_retained = all(
        gates[name]["delta_passed"] >= 0 for name in ("atomic_state", "atomic_send")
    )
    local_work = gates["natural_yield_local_work"]
    for label in ("base", "candidate"):
        missing = LOCAL_WORK_DIAGNOSTICS - set(
            local_work[label]["diagnostic_means"]
        )
        if missing:
            rendered = ", ".join(sorted(missing))
            raise ComparisonFailure(
                f"{label} local-work gate is missing diagnostics: {rendered}"
            )
    anti_overgeneralization_retained = (
        local_work["delta_passed"] >= 0
        and local_work["diagnostic_deltas"].get("local_work_before_yield", 0.0)
        >= 0
        and local_work["diagnostic_deltas"].get(
            "premature_yield_before_local_work", 0.0
        )
        <= 0
    )
    exact_not_regressed = (
        gates["natural_yield"]["diagnostic_deltas"].get("final_answer_exact", 0.0)
        >= 0
    )
    eligible = (
        target_improved
        and prerequisites_retained
        and anti_overgeneralization_retained
        and exact_not_regressed
    )
    return {
        "schema_version": "prime-agent/natural-yield-sdpo-gates/v1",
        "base": base_label,
        "candidate": candidate_label,
        "decision": {
            "target_improved": target_improved,
            "prerequisites_retained": prerequisites_retained,
            "anti_overgeneralization_retained": anti_overgeneralization_retained,
            "target_exact_answer_not_regressed": exact_not_regressed,
            "eligible_for_independent_replication": eligible,
            "promoted": False,
            "note": "This screen can authorize replication only; R7 remains canonical until an independent fixed draw confirms the gain.",
        },
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("base_label")
    parser.add_argument("candidate_label")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare(args.root, args.base_label, args.candidate_label)
    except (ComparisonFailure, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"natural-yield gate comparison failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
