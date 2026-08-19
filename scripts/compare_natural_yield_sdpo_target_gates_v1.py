"""Compare cumulative SDPO checkpoints on the frozen natural-yield target gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.compare_natural_yield_sdpo_gates_v1 import (
    MATERIAL_FORBIDDEN_TRANSITION_REDUCTION,
    TARGET_DIAGNOSTICS,
    ComparisonFailure,
    _read_gate,
)

TARGET_SUFFIX = "natural-yield"


def compare_target(
    root: Path, base_label: str, candidate_label: str
) -> dict[str, Any]:
    base = _read_gate(root, base_label, TARGET_SUFFIX)
    candidate = _read_gate(root, candidate_label, TARGET_SUFFIX)
    if base["task_signature"] != candidate["task_signature"]:
        raise ComparisonFailure("natural_yield task specifications differ")
    if base["config_signature"] != candidate["config_signature"]:
        raise ComparisonFailure("natural_yield resolved evaluation configs differ")
    for label, gate in (("base", base), ("candidate", candidate)):
        missing = TARGET_DIAGNOSTICS - set(gate["diagnostic_means"])
        if missing:
            rendered = ", ".join(sorted(missing))
            raise ComparisonFailure(
                f"{label} natural-yield gate is missing diagnostics: {rendered}"
            )

    delta_passed = candidate["passed"] - base["passed"]
    diagnostic_keys = sorted(
        set(base["diagnostic_means"]) | set(candidate["diagnostic_means"])
    )
    diagnostic_deltas = {
        key: candidate["diagnostic_means"].get(key, 0.0)
        - base["diagnostic_means"].get(key, 0.0)
        for key in diagnostic_keys
    }
    hard_improved = delta_passed > 0
    forbidden_transition_reduced_materially = (
        diagnostic_deltas["forbidden_post_spawn_tool_before_child"]
        <= -MATERIAL_FORBIDDEN_TRANSITION_REDUCTION
    )
    exact_not_regressed = diagnostic_deltas.get("final_answer_exact", 0.0) >= 0
    target_improved = hard_improved or forbidden_transition_reduced_materially
    return {
        "schema_version": "prime-agent/natural-yield-sdpo-target-gate/v1",
        "base": base_label,
        "candidate": candidate_label,
        "decision": {
            "target_improved": target_improved,
            "target_hard_improved": hard_improved,
            "target_forbidden_transition_reduced_materially": (
                forbidden_transition_reduced_materially
            ),
            "target_exact_answer_not_regressed": exact_not_regressed,
            "eligible_for_retention_gates": target_improved and exact_not_regressed,
            "promoted": False,
            "note": "This target-only screen can authorize retention gates, not replication or promotion.",
        },
        "gate": {
            "base": {key: value for key, value in base.items() if key != "task_signature"},
            "candidate": {
                key: value
                for key, value in candidate.items()
                if key != "task_signature"
            },
            "delta_passed": delta_passed,
            "diagnostic_deltas": diagnostic_deltas,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("base_label")
    parser.add_argument("candidate_label")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare_target(args.root, args.base_label, args.candidate_label)
    except (ComparisonFailure, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(
            f"natural-yield target comparison failed: {error}"
        ) from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
