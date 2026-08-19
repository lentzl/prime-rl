"""Apply the predeclared behavioral and distributional teacher admission gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.validate_prime_agent_sdpo_zero_lr_audit_v1 import _read_json


class TeacherAdmissionFailure(ValueError):
    """The paired teacher admission evidence is incomplete or inconsistent."""


def decide(distribution: dict[str, Any], behavior: dict[str, Any]) -> dict[str, Any]:
    if distribution.get("verdict") != "pass" or behavior.get("verdict") != "pass":
        raise TeacherAdmissionFailure("both audit mechanisms must validate before admission")
    if distribution.get("model_artifacts_written") is not False:
        raise TeacherAdmissionFailure("teacher admission must remain a zero-update audit")
    diversity = distribution.get("state_diversity", {})
    distribution_summary = distribution.get("distribution", {}).get("summary", {})
    behavior_summary = behavior.get("summary", {})
    states = diversity.get("distinct_states")
    if not isinstance(states, int) or states < 8 or behavior_summary.get("states") != states:
        raise TeacherAdmissionFailure("distribution and behavioral audits must share at least eight states")

    conditioned_rate = behavior_summary.get("yield_rates", {}).get("conditioned")
    absolute_gain = behavior_summary.get("conditioned_absolute_yield_gain")
    distributed_gain = behavior_summary.get(
        "states_with_conditioned_yield_and_forbidden_tool_reduction"
    )
    positive_shift_rate = distribution_summary.get("positive_away_from_tool_shift_rate")
    mean_shift = distribution_summary.get("mean_away_from_tool_log_odds_shift")
    numeric = (conditioned_rate, absolute_gain, distributed_gain, positive_shift_rate, mean_shift)
    if not all(isinstance(value, int | float) for value in numeric):
        raise TeacherAdmissionFailure("teacher admission summaries are incomplete")

    behavioral_pass = (
        conditioned_rate >= 0.30
        and absolute_gain >= 0.20
        and distributed_gain >= states / 2
        and behavior_summary.get("behavioral_teacher_admitted") is True
    )
    distributional_pass = (
        positive_shift_rate >= 0.75
        and mean_shift > 0
        and distribution_summary.get("teacher_signal_present") is True
    )
    admitted = behavioral_pass and distributional_pass
    return {
        "verdict": "pass",
        "mechanism": "natural-yield-teacher-admission-decision",
        "states": states,
        "thresholds": {
            "conditioned_valid_yield_rate_min": 0.30,
            "absolute_valid_yield_gain_min": 0.20,
            "states_with_distributed_gain_min": states // 2,
            "positive_away_from_tool_shift_rate_min": 0.75,
            "mean_away_from_tool_log_odds_shift_strictly_positive": True,
        },
        "observed": {
            "conditioned_valid_yield_rate": conditioned_rate,
            "absolute_valid_yield_gain": absolute_gain,
            "states_with_distributed_gain": distributed_gain,
            "positive_away_from_tool_shift_rate": positive_shift_rate,
            "mean_away_from_tool_log_odds_shift": mean_shift,
        },
        "behavioral_pass": behavioral_pass,
        "distributional_pass": distributional_pass,
        "teacher_update_authorized": admitted,
        "next_action": (
            "consider_one_bounded_update_from_canonical_r7"
            if admitted
            else "change_teacher_source_or_representation_without_gradient"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distribution_audit", type=Path)
    parser.add_argument("behavioral_audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = decide(_read_json(args.distribution_audit), _read_json(args.behavioral_audit))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
