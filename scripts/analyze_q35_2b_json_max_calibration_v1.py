#!/usr/bin/env python3
"""Summarize matched JSON-max direct/composed evaluation traces."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_traces(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        envelope = json.loads(line)
        traces = envelope.get("traces")
        if not isinstance(traces, list) or len(traces) != 1:
            raise ValueError(f"expected one trace per row in {path}")
        rows.append(traces[0])
    return rows


def _child_reports(trace: dict[str, Any]) -> list[str]:
    reports = []
    for node in trace["nodes"]:
        message = node.get("message", {})
        content = message.get("content")
        if message.get("role") != "tool" or not isinstance(content, str):
            continue
        try:
            parsed = ast.literal_eval(content)
        except (SyntaxError, ValueError):
            continue
        if (
            isinstance(parsed, dict)
            and parsed.get("source") == "agent_message"
            and isinstance(parsed.get("from"), dict)
            and parsed["from"].get("runtimeKind") == "subagent"
            and isinstance(parsed.get("message"), str)
        ):
            reports.append(parsed["message"])
    return reports


def _score(trace: dict[str, Any]) -> float:
    return float(trace["metrics"]["harness_score"])


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "passed": sum(_score(row) == 1.0 for row in rows),
        "pass_rate": sum(_score(row) for row in rows) / len(rows),
        "stop_conditions": dict(sorted(Counter(row["stop_condition"] for row in rows).items())),
        "mean_required_atoms_fraction": sum(
            float(row["metrics"]["required_atoms_fraction"]) for row in rows
        )
        / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    direct = _load_traces(args.result_dir / "json_max_direct_raw" / "traces.jsonl")
    composed = _load_traces(args.result_dir / "json_max_two_shard" / "traces.jsonl")
    if len(direct) != len(composed):
        raise ValueError("matched arms have different episode counts")

    direct_by_index = {int(row["task"]["data"]["idx"]): row for row in direct}
    composed_by_index = {int(row["task"]["data"]["idx"]): row for row in composed}
    if direct_by_index.keys() != composed_by_index.keys():
        raise ValueError("matched arms have different task indices")

    report_attempts = 0
    report_exact = 0
    multiple_reports = 0
    composed_child_completed = 0
    composed_child_local_reward = 0.0
    paired = []
    for index in sorted(direct_by_index):
        direct_row = direct_by_index[index]
        composed_row = composed_by_index[index]
        direct_data = direct_row["task"]["data"]
        composed_data = composed_row["task"]["data"]
        if direct_data["oracle"]["final_answer"] != composed_data["oracle"]["final_answer"]:
            raise ValueError(f"oracle answer differs at index {index}")
        if direct_data["workspace_files"] != (
            composed_data["workspace_files"]
            | composed_data["oracle"]["private_resources"]
        ):
            raise ValueError(f"raw evidence differs at index {index}")

        reports = _child_reports(composed_row)
        expected = str(composed_data["oracle"]["children"][0]["expected_result"])
        report_attempts += bool(reports)
        report_exact += bool(reports) and reports[-1].strip() == expected
        multiple_reports += len(reports) > 1
        composed_child_completed += composed_row["metrics"]["child_action_completed"] == 1.0
        composed_child_local_reward += float(
            composed_row["metrics"]["child_action_local_reward"]
        )
        paired.append(
            {
                "index": index,
                "direct_pass": _score(direct_row) == 1.0,
                "composed_pass": _score(composed_row) == 1.0,
                "child_report_observed": bool(reports),
                "child_report_exact": bool(reports) and reports[-1].strip() == expected,
                "child_contract_completed": composed_row["metrics"][
                    "child_action_completed"
                ]
                == 1.0,
            }
        )

    result = {
        "schema_version": "q35-2b-json-max-calibration-analysis/v1",
        "status": "complete",
        "direct": _summary(direct),
        "composed": {
            **_summary(composed),
            "child_report_attempts": report_attempts,
            "child_report_exact": report_exact,
            "child_contract_completed": composed_child_completed,
            "mean_child_local_reward": composed_child_local_reward / len(composed),
            "multiple_child_reports": multiple_reports,
        },
        "paired": {
            "episodes": len(paired),
            "direct_only_pass": sum(row["direct_pass"] and not row["composed_pass"] for row in paired),
            "composed_only_pass": sum(row["composed_pass"] and not row["direct_pass"] for row in paired),
            "both_pass": sum(row["direct_pass"] and row["composed_pass"] for row in paired),
            "both_fail": sum(not row["direct_pass"] and not row["composed_pass"] for row in paired),
        },
        "by_case": {
            str(case): {
                "episodes": sum(row["index"] % 5 == case for row in paired),
                "direct_pass": sum(
                    row["index"] % 5 == case and row["direct_pass"] for row in paired
                ),
                "composed_pass": sum(
                    row["index"] % 5 == case and row["composed_pass"] for row in paired
                ),
                "child_report_exact": sum(
                    row["index"] % 5 == case and row["child_report_exact"]
                    for row in paired
                ),
            }
            for case in range(5)
        },
        "rows": paired,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
