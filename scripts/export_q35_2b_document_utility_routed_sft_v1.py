#!/usr/bin/env python3
"""Build utility-topology SFT rows aligned to the routed root prompt."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset
from dual_policy_openai_proxy_v1 import (
    DOCUMENT_ROOT_CAUSAL_UTILITY_DECISION_CONTRACT,
    DOCUMENT_ROOT_UTILITY_DECISION_CONTRACT,
    ROOT_COORDINATOR_CAUSAL_UTILITY_DECISION_CONTRACT,
    ROOT_COORDINATOR_CONTRACT,
    ROOT_COORDINATOR_UTILITY_DECISION_CONTRACT,
)
from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_document_utility_topology_sft_v1 import (
    FAMILY_TO_TOPOLOGY_FAMILY,
    _utility_row,
)

SCHEMA_VERSION = "qwen35-2b-document-utility-routed-consolidated-sft/v2"
OBJECTIVE = (
    "answer_free_root_topology_choice_from_routed_ownership_and_resource_constraints"
)
DIRECT_SCHEMA_VERSION = "qwen35-2b-document-utility-routed-direct-consolidated-sft/v2"
DIRECT_OBJECTIVE = "answer_free_root_direct_utility_choice_from_routed_constraints"
EXPANDED_SCHEMA_VERSION = (
    "qwen35-2b-document-utility-routed-expanded-consolidated-sft/v2"
)
EXPANDED_OBJECTIVE = (
    "answer_free_root_topology_choice_from_expanded_routed_ownership_and_resource_constraints"
)
RUBRIC_SCHEMA_VERSION = (
    "qwen35-2b-document-utility-routed-rubric-expanded-consolidated-sft/v2"
)
RUBRIC_OBJECTIVE = (
    "answer_free_root_topology_choice_from_routed_resource_decision_rubric"
)
DIRECT_FAMILY = "document_utility_direct"
ROWS = 24
PER_FAMILY = 8
EXPANDED_ROWS = 36
EXPANDED_PER_FAMILY = 12
CAUSAL_SCHEMA_VERSION = (
    "qwen35-2b-document-utility-routed-causal-matched-consolidated-sft/v1"
)
CAUSAL_BALANCED_SCHEMA_VERSION = (
    "qwen35-2b-document-utility-routed-causal-matched-balanced-consolidated-sft/v1"
)
CAUSAL_OBJECTIVE = "answer_free_causal_matched_root_topology_choice"
HISTORICAL_POLICIES = {
    "document_utility_direct": (
        "The root is permitted to inspect the directory. The total agent-admission "
        "budget is zero. Select the reliable plan that satisfies that budget."
    ),
    "document_utility_flat": (
        "The root is not permitted to inspect the directory. Descendant depth is "
        "limited to one, and the root may admit at most three agents. Select the "
        "reliable plan that respects ownership and depth."
    ),
    "document_utility_hierarchical": (
        "The root is not permitted to inspect the directory and may admit at most "
        "one agent. That admitted agent may delegate at one further depth. Select "
        "the reliable plan that respects ownership and both budgets."
    ),
}
CAUSAL_POLICIES = {
    "document_utility_direct": (
        "The root is permitted to inspect the directory and may admit up to three "
        "agents. An admitted coordinator may delegate at one further depth. All "
        "legal plans are otherwise equally reliable. Select the plan with the "
        "fewest total agent admissions."
    ),
    "document_utility_flat": (
        "The root is not permitted to inspect the directory and may admit up to "
        "three agents. An admitted coordinator may delegate at one further depth. "
        "All legal plans are otherwise equally reliable. Select the plan with the "
        "fewest total agent admissions."
    ),
    "document_utility_hierarchical": (
        "The root is not permitted to inspect the directory and may admit at most "
        "one agent. An admitted coordinator may delegate at one further depth. All "
        "legal plans are otherwise equally reliable. Select the plan with the "
        "fewest total agent admissions."
    ),
}


def _sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    match = re.search(r"-v(?P<variant>\d+)-i(?P<instance>\d+)$", row["task_key"])
    if match is None:
        raise ValueError(f"invalid routed utility task key: {row['task_key']!r}")
    return int(match.group("instance")), int(match.group("variant")), row["family"]


def _replace_message_text(message: dict[str, Any], old: str, new: str) -> int:
    content = message.get("content")
    if isinstance(content, str):
        count = content.count(old)
        message["content"] = content.replace(old, new)
        return count
    if isinstance(content, list):
        count = 0
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                count += part["text"].count(old)
                part["text"] = part["text"].replace(old, new)
        return count
    return 0


def _causalized_trace(trace: dict[str, Any]) -> dict[str, Any]:
    transformed = copy.deepcopy(trace)
    task = transformed.get("task", {}).get("data", {})
    family = task.get("family")
    old = HISTORICAL_POLICIES.get(family)
    new = CAUSAL_POLICIES.get(family)
    if old is None or new is None:
        raise ValueError(f"cannot causalize utility family: {family!r}")
    prompt = task.get("prompt")
    if not isinstance(prompt, str) or prompt.count(old) != 1:
        raise ValueError("causal utility source prompt lacks one historical policy")
    task["prompt"] = prompt.replace(old, new)
    task["utility_policy_profile"] = "causal_matched_v2"
    replacements = sum(
        _replace_message_text(node.get("message") or {}, old, new)
        for node in transformed.get("nodes") or []
        if isinstance(node, dict)
    )
    if replacements != 1:
        raise ValueError(
            f"causal utility trace must rewrite one wire prompt, rewrote {replacements}"
        )
    return transformed


def export(
    *,
    traces: list[Path],
    output_dir: Path,
    focus_family: str | None = None,
    expanded: bool = False,
    utility_rubric: bool = False,
    causal_matched: bool = False,
    flat_repeat: int = 1,
) -> dict[str, Any]:
    if focus_family not in {None, DIRECT_FAMILY}:
        raise ValueError(f"unsupported routed utility focus: {focus_family!r}")
    if focus_family is not None and expanded:
        raise ValueError("expanded routed utility export cannot focus one family")
    if utility_rubric and (not expanded or focus_family is not None):
        raise ValueError("utility rubric export requires the expanded balanced bank")
    if causal_matched and not utility_rubric:
        raise ValueError("causal matched export requires the utility rubric")
    if flat_repeat < 1 or (flat_repeat != 1 and not causal_matched):
        raise ValueError("flat repeat must be positive and is only valid for causal export")
    selected_families = (
        {focus_family} if focus_family else set(FAMILY_TO_TOPOLOGY_FAMILY)
    )
    if causal_matched:
        schema_version = (
            CAUSAL_BALANCED_SCHEMA_VERSION
            if flat_repeat == 1
            else CAUSAL_SCHEMA_VERSION
        )
        objective = CAUSAL_OBJECTIVE
    elif utility_rubric:
        schema_version, objective = RUBRIC_SCHEMA_VERSION, RUBRIC_OBJECTIVE
    elif focus_family:
        schema_version, objective = DIRECT_SCHEMA_VERSION, DIRECT_OBJECTIVE
    elif expanded:
        schema_version, objective = EXPANDED_SCHEMA_VERSION, EXPANDED_OBJECTIVE
    else:
        schema_version, objective = SCHEMA_VERSION, OBJECTIVE
    per_family = EXPANDED_PER_FAMILY if expanded else PER_FAMILY
    expected_rows = per_family if focus_family else (EXPANDED_ROWS if expanded else ROWS)
    if causal_matched:
        expected_rows += (flat_repeat - 1) * EXPANDED_PER_FAMILY
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite routed document utility bootstrap: {output_dir}"
        )
    rows: list[dict[str, Any]] = []
    sources = []
    for path in traces:
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        sources.append({"path": str(resolved), "sha256": sha256_file(path)})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                for trace in json.loads(line).get("traces") or []:
                    family = trace.get("task", {}).get("data", {}).get("family")
                    if family not in selected_families:
                        continue
                    source_trace = _causalized_trace(trace) if causal_matched else trace
                    row = _utility_row(source_trace, source=resolved)
                    if any(
                        message.get("content") == ROOT_COORDINATOR_CONTRACT
                        for message in row["messages"]
                    ):
                        raise ValueError("source trace already contains the routed root contract")
                    root_contract = ROOT_COORDINATOR_CONTRACT
                    if causal_matched:
                        root_contract = ROOT_COORDINATOR_CAUSAL_UTILITY_DECISION_CONTRACT
                    elif utility_rubric:
                        root_contract = ROOT_COORDINATOR_UTILITY_DECISION_CONTRACT
                    prime_instruction = row["messages"][0]
                    if (
                        prime_instruction.get("role") != "user"
                        or not isinstance(prime_instruction.get("content"), str)
                        or not prime_instruction["content"].strip()
                    ):
                        raise ValueError(
                            "routed utility row lacks the Prime Agent instruction prefix"
                        )
                    row["messages"][0] = {
                        "role": "system",
                        "content": (
                            f"{root_contract}\n\n{prime_instruction['content'].strip()}"
                        ),
                    }
                    row["objective"] = objective
                    rows.append(row)
    rows.sort(key=_sort_key)
    if causal_matched and flat_repeat > 1:
        expanded_rows = []
        for row in rows:
            expanded_rows.append(row)
            if row["family"] == "document_utility_flat":
                for repeat in range(2, flat_repeat + 1):
                    repeated = copy.deepcopy(row)
                    repeated["task_key"] = f"{row['task_key']}#flat-repeat-{repeat}"
                    repeated["trace_id"] = f"{row['trace_id']}#flat-repeat-{repeat}"
                    expanded_rows.append(repeated)
        rows = expanded_rows
    if len(rows) != expected_rows or len({row["task_key"] for row in rows}) != expected_rows:
        raise ValueError(
            f"routed document utility bootstrap requires {expected_rows} unique rows"
        )
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted(selected_families)
    }
    expected_family_counts = {
        family: (
            per_family * flat_repeat
            if causal_matched and family == "document_utility_flat"
            else per_family
        )
        for family in sorted(selected_families)
    }
    if family_counts != expected_family_counts:
        raise ValueError(
            f"routed document utility bootstrap is not balanced: {family_counts}"
        )
    if focus_family is None and not causal_matched:
        for start in range(0, expected_rows, 6):
            if {row["family"] for row in rows[start : start + 6]} != set(
                FAMILY_TO_TOPOLOGY_FAMILY
            ):
                raise ValueError("routed document utility ordering is not locally balanced")
    system_prefix_hashes = {
        hashlib.sha256(row["messages"][0]["content"].encode()).hexdigest()
        for row in rows
    }
    system_prefix_set_sha256 = hashlib.sha256(
        "\n".join(sorted(system_prefix_hashes)).encode()
    ).hexdigest()

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": schema_version,
        "status": "complete",
        "role": "coordinator",
        "objective": objective,
        "rows": expected_rows,
        "training_batch_size": 8 if focus_family else 12,
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": sources,
        "answer_free": True,
        "topology_choice_targeted": True,
        "utility_constraints_targeted": True,
        "root_coordinator_contract_aligned": True,
        "utility_decision_rubric_aligned": utility_rubric,
        "utility_policy_profile": (
            "causal_matched_v2" if causal_matched else "historical_v1"
        ),
        "causal_one_fact_pairs": causal_matched,
        "flat_repeat": flat_repeat,
        "utility_decision_rubric_serialization": (
            "vllm_developer_consolidated_system_v1" if utility_rubric else None
        ),
        "vllm_developer_system_consolidation_aligned": True,
        "prime_agent_instruction_consolidated": True,
        "live_system_prefix_count": len(system_prefix_hashes),
        "live_system_prefix_set_sha256": system_prefix_set_sha256,
        "expanded_prompt_bank": expanded,
        "remedial_classes": sorted(selected_families) if focus_family else [],
        "root_coordinator_contract_sha256": hashlib.sha256(
            ROOT_COORDINATOR_CONTRACT.encode()
        ).hexdigest(),
        "utility_decision_rubric_sha256": (
            hashlib.sha256(
                (
                    DOCUMENT_ROOT_CAUSAL_UTILITY_DECISION_CONTRACT
                    if causal_matched
                    else DOCUMENT_ROOT_UTILITY_DECISION_CONTRACT
                ).encode()
            ).hexdigest()
            if utility_rubric
            else None
        ),
        "combined_system_contract_sha256": (
            hashlib.sha256(
                (
                    ROOT_COORDINATOR_CAUSAL_UTILITY_DECISION_CONTRACT
                    if causal_matched
                    else ROOT_COORDINATOR_UTILITY_DECISION_CONTRACT
                ).encode()
            ).hexdigest()
            if utility_rubric
            else None
        ),
        "leading_system_message_count": 1,
        "protocol_mechanics_targeted": False,
        "tool_call_format": "openai_function_v1",
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-family", choices=[DIRECT_FAMILY])
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--utility-rubric", action="store_true")
    parser.add_argument("--causal-matched", action="store_true")
    parser.add_argument("--flat-repeat", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                traces=[path.resolve() for path in args.traces],
                output_dir=args.output_dir.resolve(),
                focus_family=args.focus_family,
                expanded=args.expanded,
                utility_rubric=args.utility_rubric,
                causal_matched=args.causal_matched,
                flat_repeat=args.flat_repeat,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
