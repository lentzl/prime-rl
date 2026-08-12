#!/usr/bin/env python3
"""Classify a frozen native-sibling continuation trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RUN265_NATURAL_PATH_ACCESSES = 4
RUN265_OWNERSHIP_GAINS = 2
RUN265_HANDSHAKE_BIDIRECTIONAL = 0.8863636363636364
FROZEN_CONTINUATION_HORIZON = 4


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summarize(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for step_dir in sorted(root.glob("step-*"), key=lambda path: int(path.name.split("-")[-1])):
        ownership_path = step_dir / "ownership-selection.json"
        natural_path = step_dir / "natural-selection.json"
        if not ownership_path.exists() or not natural_path.exists():
            continue
        ownership = _load(ownership_path)
        natural = _load(natural_path)
        natural_path_accesses = sum(
            values["coordinator_delegated_path_accesses"] for values in natural["candidate"].values()
        )
        natural_answers = sum(values["answer_accuracy"] for values in natural["candidate"].values())
        hard_invariants = (
            ownership["candidate_child_path_accesses"] == 0
            and ownership["candidate_direct_spawns"] == 0
            and natural_path_accesses <= RUN265_NATURAL_PATH_ACCESSES
            and natural_answers == 8
        )
        followup = natural["candidate"]["followup"]
        handshake = natural["candidate"]["handshake"]
        rows.append(
            {
                "additional_updates": int(step_dir.name.split("-")[-1]),
                "ownership_promotion_pass": ownership["promotion_pass"],
                "natural_promotion_pass": natural["promotion_pass"],
                "promotion_pass": ownership["promotion_pass"] and natural["promotion_pass"],
                "strict_gains": ownership["strict_gains"],
                "strict_losses": ownership["strict_losses"],
                "candidate_child_path_accesses": ownership["candidate_child_path_accesses"],
                "candidate_direct_spawns": ownership["candidate_direct_spawns"],
                "natural_path_accesses": natural_path_accesses,
                "natural_answers": natural_answers,
                "followup_causal": followup["natural_followup_causal"],
                "followup_protocol": followup["protocol_aligned"],
                "handshake_causal": handshake["natural_followup_causal"],
                "handshake_protocol": handshake["protocol_aligned"],
                "handshake_bidirectional": handshake["bidirectional_control_mean"],
                "hard_invariants_pass": hard_invariants,
                "followup_recovered": followup["natural_followup_causal"] >= 4,
                "handshake_gain_retained": (
                    handshake["protocol_aligned"] >= 4
                    and handshake["bidirectional_control_mean"] >= RUN265_HANDSHAKE_BIDIRECTIONAL
                ),
                "ownership_broadened": ownership["strict_gains"] > RUN265_OWNERSHIP_GAINS,
            }
        )

    promotion = next((row for row in rows if row["promotion_pass"]), None)
    hard_failure = next((row for row in rows if not row["hard_invariants_pass"]), None)
    consolidation = next(
        (
            row
            for row in rows
            if row["followup_recovered"] and row["handshake_gain_retained"] and row["ownership_broadened"]
        ),
        None,
    )
    if promotion is not None:
        classification = "PROMOTE_CANDIDATE"
    elif hard_failure is not None:
        classification = "BRANCH_REJECTED_HARD_INVARIANT"
    elif consolidation is not None:
        classification = "CONSOLIDATION_EVIDENCE"
    elif (
        len(rows) >= FROZEN_CONTINUATION_HORIZON
        and rows[-1]["ownership_broadened"]
        and not rows[-1]["followup_recovered"]
    ):
        classification = "STABLE_TRADEOFF_EVIDENCE"
    else:
        classification = "CONTINUATION_INCONCLUSIVE"

    return {
        "classification": classification,
        "evaluated_checkpoints": len(rows),
        "frozen_continuation_horizon": FROZEN_CONTINUATION_HORIZON,
        "trajectory_complete": len(rows) >= FROZEN_CONTINUATION_HORIZON,
        "first_promotion_update": promotion["additional_updates"] if promotion else None,
        "first_hard_failure_update": hard_failure["additional_updates"] if hard_failure else None,
        "first_consolidation_update": consolidation["additional_updates"] if consolidation else None,
        "frozen_references": {
            "run265_natural_path_accesses": RUN265_NATURAL_PATH_ACCESSES,
            "run265_ownership_gains": RUN265_OWNERSHIP_GAINS,
            "run265_handshake_bidirectional": RUN265_HANDSHAKE_BIDIRECTIONAL,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
