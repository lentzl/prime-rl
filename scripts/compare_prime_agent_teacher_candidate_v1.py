"""Compare untouched and updated Prime Agent mastery and resilience summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_COUNTS = {"mastery": 74, "resilience": 12}
FOUNDATION_FAMILIES = {
    "child_cancellation",
    "child_result_delivery",
    "conversation_resume",
    "ipython_cell",
    "kernel_persistence",
}
TARGET_FAMILIES = {
    "child",
    "single",
    "parallel",
    "followup",
    "handshake",
    "malformed_result_repair",
    "delayed_result",
    "message_type_repair",
}


class ComparisonFailure(ValueError):
    """The summaries cannot support a candidate comparison."""


def _read_summary(path: Path, expected_count: int) -> dict[str, Any]:
    if not path.is_file():
        raise ComparisonFailure(f"missing summary: {path}")
    summary = json.loads(path.read_text())
    if not isinstance(summary, dict) or summary.get("trace_count") != expected_count:
        raise ComparisonFailure(f"summary at {path} does not contain {expected_count} traces")
    if not isinstance(summary.get("families"), dict) or not isinstance(summary.get("tasks"), list):
        raise ComparisonFailure(f"summary at {path} has an invalid schema")
    return summary


def _task_map(summary: dict[str, Any], suite: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    tasks = {}
    for task in summary["tasks"]:
        key = (suite, str(task.get("family")), str(task.get("name")))
        if key in tasks:
            raise ComparisonFailure(f"duplicate task identity: {key}")
        tasks[key] = task
    return tasks


def _family_deltas(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if set(base["families"]) != set(candidate["families"]):
        raise ComparisonFailure("base and candidate family sets differ")
    deltas = {}
    for family in sorted(base["families"]):
        base_family = base["families"][family]
        candidate_family = candidate["families"][family]
        if base_family.get("count") != candidate_family.get("count"):
            raise ComparisonFailure(f"base and candidate counts differ for {family}")
        metric_names = set(base_family.get("means", {})) | set(candidate_family.get("means", {}))
        metric_deltas = {
            metric: candidate_family.get("means", {}).get(metric, 0.0)
            - base_family.get("means", {}).get(metric, 0.0)
            for metric in sorted(metric_names)
        }
        deltas[family] = {
            "count": base_family["count"],
            "clean_count": candidate_family["clean_count"] - base_family["clean_count"],
            "mean_reward": candidate_family["mean_reward"] - base_family["mean_reward"],
            "means": metric_deltas,
        }
    return deltas


def compare(
    base_mastery: Path,
    candidate_mastery: Path,
    base_resilience: Path,
    candidate_resilience: Path,
) -> dict[str, Any]:
    summaries = {
        "base_mastery": _read_summary(base_mastery, EXPECTED_COUNTS["mastery"]),
        "candidate_mastery": _read_summary(candidate_mastery, EXPECTED_COUNTS["mastery"]),
        "base_resilience": _read_summary(base_resilience, EXPECTED_COUNTS["resilience"]),
        "candidate_resilience": _read_summary(candidate_resilience, EXPECTED_COUNTS["resilience"]),
    }
    base_tasks = {
        **_task_map(summaries["base_mastery"], "mastery"),
        **_task_map(summaries["base_resilience"], "resilience"),
    }
    candidate_tasks = {
        **_task_map(summaries["candidate_mastery"], "mastery"),
        **_task_map(summaries["candidate_resilience"], "resilience"),
    }
    if set(base_tasks) != set(candidate_tasks):
        missing = sorted(set(base_tasks) - set(candidate_tasks))
        extra = sorted(set(candidate_tasks) - set(base_tasks))
        raise ComparisonFailure(f"task identities differ: missing={missing[:3]} extra={extra[:3]}")

    gains = []
    losses = []
    for key in sorted(base_tasks):
        base_clean = not base_tasks[key].get("issues")
        candidate_clean = not candidate_tasks[key].get("issues")
        record = {"suite": key[0], "family": key[1], "name": key[2]}
        if not base_clean and candidate_clean:
            gains.append(record)
        elif base_clean and not candidate_clean:
            losses.append(record)

    family_deltas = {
        "mastery": _family_deltas(
            summaries["base_mastery"], summaries["candidate_mastery"]
        ),
        "resilience": _family_deltas(
            summaries["base_resilience"], summaries["candidate_resilience"]
        ),
    }
    hard_rejections = []
    protected_losses = [
        loss
        for loss in losses
        if loss["family"] in FOUNDATION_FAMILIES or loss["family"] == "direct"
    ]
    if protected_losses:
        hard_rejections.append("foundation_or_direct_clean_loss")

    child_delta = family_deltas["mastery"].get("child", {}).get("means", {})
    if child_delta.get("parent_path_access", 0.0) > 1e-9:
        hard_rejections.append("child_owned_path_leakage_increased")
    coordinator_delta = family_deltas["mastery"].get("coordinator", {}).get("means", {})
    if coordinator_delta.get("direct_answer_accuracy", 0.0) < -1e-9:
        hard_rejections.append("coordinator_owned_control_regressed")
    if family_deltas["mastery"].get("oolong", {}).get("mean_reward", 0.0) < -0.1:
        hard_rejections.append("oolong_reward_regressed_broadly")

    target_gains = [gain for gain in gains if gain["family"] in TARGET_FAMILIES]
    target_losses = [loss for loss in losses if loss["family"] in TARGET_FAMILIES]
    all_candidate_clean = all(
        summary["issue_count"] == 0
        for name, summary in summaries.items()
        if name.startswith("candidate_")
    )
    if all_candidate_clean and not hard_rejections:
        verdict = "PROMOTION-ELIGIBLE-PENDING-INDEPENDENT-REPEAT"
    elif (
        not hard_rejections
        and len(target_gains) >= 2
        and len(target_gains) > len(target_losses)
    ):
        verdict = "CONTINUATION-ELIGIBLE"
    else:
        verdict = "BRANCH-REJECTED"

    return {
        "schema_version": 1,
        "verdict": verdict,
        "task_count": len(base_tasks),
        "clean_gains": gains,
        "clean_losses": losses,
        "target_clean_gains": len(target_gains),
        "target_clean_losses": len(target_losses),
        "hard_rejections": hard_rejections,
        "family_deltas": family_deltas,
        "candidate_issue_counts": {
            "mastery": summaries["candidate_mastery"]["issue_count"],
            "resilience": summaries["candidate_resilience"]["issue_count"],
        },
        "note": (
            "Promotion eligibility still requires an independent repeated screen; "
            "this report does not declare a canonical teacher."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-mastery", type=Path, required=True)
    parser.add_argument("--candidate-mastery", type=Path, required=True)
    parser.add_argument("--base-resilience", type=Path, required=True)
    parser.add_argument("--candidate-resilience", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare(
            args.base_mastery,
            args.candidate_mastery,
            args.base_resilience,
            args.candidate_resilience,
        )
    except (ComparisonFailure, json.JSONDecodeError) as error:
        raise SystemExit(f"teacher-candidate comparison failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
