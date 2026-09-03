#!/usr/bin/env python3
"""Build causally matched SFT for public-registry expert selection."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset
from dual_policy_openai_proxy_v1 import specialist_manager_contract_from_messages
from export_q35_2b_adaptive_cognition_sft_v1 import _runtime_messages
from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_specialist_expert_sft_v1 import EXPERT_IDS, _row
from subagent_communication_v1.taskset import (
    SPECIALIST_EXPERTS,
    SPECIALIST_FAMILIES,
    SubagentCommunicationConfig,
    SubagentCommunicationTaskset,
)

SCHEMA_VERSION = "qwen35-2b-specialist-router-contrast-sft/v1"
OBJECTIVE = "answer_free_causal_registry_capability_matching"
TRAINING_INSTANCE_OFFSET = 37900
TRAINING_SEED = 20261211
BASE_GROUPS = 16
PERMUTATIONS_PER_GROUP = 6
ROWS = BASE_GROUPS * PERMUTATIONS_PER_GROUP
ROWS_PER_EXPERT = ROWS // len(EXPERT_IDS)


def _permuted_registry_prompt(
    prompt: str, *, profile_by_expert_id: dict[str, str]
) -> str:
    if set(profile_by_expert_id) != set(EXPERT_IDS) or set(
        profile_by_expert_id.values()
    ) != set(EXPERT_IDS):
        raise ValueError("registry intervention must be a capability permutation")
    rewrites = 0
    lines = []
    for line in prompt.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            lines.append(line)
            continue
        expert_id = entry.get("expert_id") if isinstance(entry, dict) else None
        if expert_id not in profile_by_expert_id or entry.get("role") != "terminal_worker":
            lines.append(line)
            continue
        profile_id = profile_by_expert_id[expert_id]
        profile = SPECIALIST_EXPERTS[profile_id]
        entry["capability"] = profile["capability"]
        entry["limitations"] = profile["limitations"]
        lines.append(json.dumps(entry, separators=(",", ":")))
        rewrites += 1
    if rewrites < len(EXPERT_IDS) or rewrites % len(EXPERT_IDS):
        raise ValueError(f"expected complete registry blocks, rewrote {rewrites} entries")
    return "\n".join(lines)


def _target_for_profile(
    profile_by_expert_id: dict[str, str], required_profile: str
) -> str:
    matches = [
        expert_id
        for expert_id, profile_id in profile_by_expert_id.items()
        if profile_id == required_profile
    ]
    if len(matches) != 1:
        raise ValueError(f"required profile is not unique: {required_profile}")
    return matches[0]


def _training_tasks() -> dict[str, list[Any]]:
    taskset = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=tuple(SPECIALIST_FAMILIES),
            instances_per_template=4,
            instance_offset=TRAINING_INSTANCE_OFFSET,
            seed=TRAINING_SEED,
            available_experts=EXPERT_IDS,
        )
    )
    grouped: dict[str, list[Any]] = {family: [] for family in SPECIALIST_FAMILIES}
    for task in taskset.load():
        grouped[task.data.family].append(task)
    for values in grouped.values():
        values.sort(key=lambda task: task.data.name)
    return grouped


def _base_groups(tasks: dict[str, list[Any]]) -> list[dict[str, Any]]:
    generic = [
        {
            "key": task.data.name,
            "prompt": task.data.prompt,
            "required_profile": "generic_worker",
            "role_scope": "root",
            "depth": 0,
            "root": True,
        }
        for task in tasks["specialist_generic"][:4]
    ]
    table_root = [
        {
            "key": task.data.name,
            "prompt": task.data.prompt,
            "required_profile": "table_analyst",
            "role_scope": "root",
            "depth": 0,
            "root": True,
        }
        for family, count in (
            ("specialist_table_join", 2),
            ("specialist_table_reconcile", 1),
        )
        for task in tasks[family][:count]
    ]
    source_root = [
        {
            "key": task.data.name,
            "prompt": task.data.prompt,
            "required_profile": "source_inspector",
            "role_scope": "root",
            "depth": 0,
            "root": True,
        }
        for family, count in (
            ("specialist_source_ast", 2),
            ("specialist_source_config", 1),
        )
        for task in tasks[family][:count]
    ]
    managers: dict[str, list[dict[str, Any]]] = {
        "table_analyst": [],
        "source_inspector": [],
    }
    for required_profile, family in (
        ("table_analyst", "specialist_recursive_table"),
        ("source_inspector", "specialist_recursive_source"),
    ):
        for task in tasks[family][:3]:
            manager = specialist_manager_contract_from_messages(
                [{"role": "user", "content": task.data.prompt}]
            )
            if manager is None:
                raise ValueError(f"recursive specialist task lacks manager: {task.data.name}")
            managers[required_profile].append(
                {
                    "key": f"{task.data.name}:specialist-manager",
                    "prompt": manager,
                    "required_profile": required_profile,
                    "role_scope": "nonroot_specialist_manager",
                    "depth": 1,
                    "root": False,
                }
            )
    categories = [
        generic,
        table_root,
        source_root,
        managers["table_analyst"],
        managers["source_inspector"],
    ]
    groups = []
    for index in range(4):
        for category in categories:
            if index < len(category):
                groups.append(category[index])
    if len(groups) != BASE_GROUPS:
        raise ValueError(f"expected {BASE_GROUPS} causal groups, found {len(groups)}")
    return groups


def _candidate_rows(
    runtime: dict[int, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups = _base_groups(_training_tasks())
    passes: dict[str, list[list[dict[str, Any]]]] = {
        expert_id: [[], []] for expert_id in EXPERT_IDS
    }
    permutations = list(itertools.permutations(EXPERT_IDS))
    if len(permutations) != PERMUTATIONS_PER_GROUP:
        raise ValueError("unexpected expert permutation count")
    for group_index, group in enumerate(groups):
        by_target: dict[str, list[dict[str, Any]]] = {
            expert_id: [] for expert_id in EXPERT_IDS
        }
        for permutation_index, profile_order in enumerate(permutations):
            profile_by_expert_id = dict(zip(EXPERT_IDS, profile_order, strict=True))
            target = _target_for_profile(
                profile_by_expert_id, group["required_profile"]
            )
            prompt = _permuted_registry_prompt(
                group["prompt"], profile_by_expert_id=profile_by_expert_id
            )
            row = _row(
                runtime=runtime[group["depth"]],
                prompt=prompt,
                key=f"{group['key']}:registry-permutation-{permutation_index}",
                depth=group["depth"],
                role_scope=group["role_scope"],
                expert_id=target,
                root=group["root"],
            )
            row.update(
                {
                    "family": "specialist_router_causal_contrast",
                    "objective": OBJECTIVE,
                    "contrast_group": group_index,
                    "required_capability_profile": group["required_profile"],
                    "profile_by_expert_id": profile_by_expert_id,
                }
            )
            by_target[target].append(row)
        for expert_id, values in by_target.items():
            if len(values) != 2:
                raise ValueError(
                    f"group {group_index} has {len(values)} rows for {expert_id}"
                )
            passes[expert_id][0].append(values[0])
            passes[expert_id][1].append(values[1])
    return {
        expert_id: [*passes[expert_id][0], *passes[expert_id][1]]
        for expert_id in EXPERT_IDS
    }


def export(*, runtime_traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite router contrast SFT: {output_dir}")
    runtime, sources = _runtime_messages(runtime_traces)
    pools = _candidate_rows(runtime)
    rows = [
        pools[expert_id][index]
        for index in range(ROWS_PER_EXPERT)
        for expert_id in EXPERT_IDS
    ]
    if len(rows) != ROWS or len({row["task_key"] for row in rows}) != ROWS:
        raise ValueError("router contrast SFT requires 96 unique rows")
    expert_counts = {
        expert_id: sum(row["expert_id"] == expert_id for row in rows)
        for expert_id in EXPERT_IDS
    }
    role_counts = {
        expert_id: {
            role: sum(
                row["expert_id"] == expert_id and row["role_scope"] == role
                for row in rows
            )
            for role in ("root", "nonroot_specialist_manager")
        }
        for expert_id in EXPERT_IDS
    }
    first_half = rows[:48]
    first_half_counts = Counter(row["expert_id"] for row in first_half)
    first_half_groups = Counter(row["contrast_group"] for row in first_half)
    expected_roles = {
        expert_id: {"root": 20, "nonroot_specialist_manager": 12}
        for expert_id in EXPERT_IDS
    }
    if (
        expert_counts != {expert_id: ROWS_PER_EXPERT for expert_id in EXPERT_IDS}
        or role_counts != expected_roles
        or first_half_counts != Counter({expert_id: 16 for expert_id in EXPERT_IDS})
        or set(first_half_groups.values()) != {3}
        or len(first_half_groups) != BASE_GROUPS
    ):
        raise ValueError("router contrast curriculum balance is invalid")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "specialist_router",
        "objective": OBJECTIVE,
        "rows": ROWS,
        "training_batch_size": 12,
        "expert_counts": expert_counts,
        "role_counts": role_counts,
        "first_half_expert_counts": dict(sorted(first_half_counts.items())),
        "first_batch_expert_counts": dict(
            sorted(Counter(row["expert_id"] for row in rows[:12]).items())
        ),
        "first_half_contrast_group_counts": {
            str(key): value for key, value in sorted(first_half_groups.items())
        },
        "base_assignment_groups": BASE_GROUPS,
        "registry_permutations_per_group": PERMUTATIONS_PER_GROUP,
        "runtime_traces": sources,
        "answer_free": True,
        "public_capability_registry_only": True,
        "expert_only_tool_arguments": True,
        "cognitive_action_labels_present": False,
        "causal_registry_permutations": True,
        "training_instance_offset": TRAINING_INSTANCE_OFFSET,
        "training_template_variants": [0, 1, 2, 3],
        "heldout_template_variants_excluded": [4, 5],
        "observed_instance_offsets_excluded": [
            35100,
            37100,
            37200,
            37300,
            37400,
            37500,
            37700,
        ],
        "tool_call_format": "openai_function_v1",
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-traces", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                runtime_traces=[path.resolve() for path in args.runtime_traces],
                output_dir=args.output_dir.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
