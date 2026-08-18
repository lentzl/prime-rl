"""Compare two checkpoints on identical procedural harness action gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from .summarize_procedural_harness_master_v1 import _traces
else:
    from summarize_procedural_harness_master_v1 import _traces

REQUIRED_RUNGS = (
    "atomic_state",
    "atomic_send",
    "atomic_child_request",
    "atomic_followup",
)


def _task_signature(trace: dict[str, Any]) -> str:
    data = trace.get("task", {}).get("data", {})
    stable = {key: data.get(key) for key in ("episode_id", "family", "prompt", "system_prompt", "oracle")}
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def _read_gate(root: Path, label: str, rung: str) -> dict[str, Any]:
    gate = root / f"{label}-{rung}-gate-r1" / "train-admission"
    summary_path = gate / "SUMMARY.json"
    if not summary_path.is_file():
        raise ValueError(f"missing action-gate summary: {summary_path}")
    report = json.loads(summary_path.read_text())
    if report.get("rescored") is not True:
        raise ValueError(f"action-gate summary was not rescored: {summary_path}")
    if report.get("episodes") != 8:
        raise ValueError(f"action gate must contain eight episodes: {summary_path}")
    if report.get("errors") != 0:
        raise ValueError(f"action gate contains rollout errors: {summary_path}")
    if set(report.get("by_family", {})) != {rung}:
        raise ValueError(f"action gate does not contain only {rung}: {summary_path}")

    traces = _traces(gate)
    if len(traces) != 8:
        raise ValueError(f"action gate must contain eight traces: {gate}")
    signatures = {_task_signature(trace) for trace in traces}
    if len(signatures) != 1:
        raise ValueError(f"action gate contains more than one task specification: {gate}")

    harness = report["harness"]
    return {
        "label": label,
        "task_signature": signatures.pop(),
        "passed": harness["passed"],
        "episodes": harness["episodes"],
        "rate": harness["rate"],
        "diagnostic_means": report.get("diagnostic_means", {}),
    }


def compare(root: Path, base_label: str, candidate_label: str) -> dict[str, Any]:
    rungs: dict[str, Any] = {}
    for rung in REQUIRED_RUNGS:
        base = _read_gate(root, base_label, rung)
        candidate = _read_gate(root, candidate_label, rung)
        if base["task_signature"] != candidate["task_signature"]:
            raise ValueError(f"{rung} checkpoints were evaluated on different task specifications")

        diagnostic_keys = sorted(set(base["diagnostic_means"]) | set(candidate["diagnostic_means"]))
        rungs[rung] = {
            "base": {key: value for key, value in base.items() if key != "task_signature"},
            "candidate": {key: value for key, value in candidate.items() if key != "task_signature"},
            "delta_passed": candidate["passed"] - base["passed"],
            "diagnostic_deltas": {
                key: candidate["diagnostic_means"].get(key, 0.0) - base["diagnostic_means"].get(key, 0.0)
                for key in diagnostic_keys
            },
        }

    target_improved = rungs["atomic_child_request"]["delta_passed"] > 0
    prerequisites_retained = all(rungs[rung]["delta_passed"] >= 0 for rung in ("atomic_state", "atomic_send"))
    followup_not_regressed = rungs["atomic_followup"]["delta_passed"] >= 0
    screen_pass = target_improved and prerequisites_retained and followup_not_regressed
    return {
        "schema_version": 1,
        "base": base_label,
        "candidate": candidate_label,
        "screen": {
            "target_improved": target_improved,
            "prerequisites_retained": prerequisites_retained,
            "followup_not_regressed": followup_not_regressed,
            "pass": screen_pass,
            "note": (
                "A pass makes the candidate eligible for an independent replication; "
                "it does not by itself establish full harness mastery."
            ),
        },
        "rungs": rungs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("base_label")
    parser.add_argument("candidate_label")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(
        compare(args.root, args.base_label, args.candidate_label),
        indent=2,
        sort_keys=True,
    )
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
