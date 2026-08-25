#!/usr/bin/env python3
"""Export answer-free bootstrap scaffolds as coordinator-side designer SFT rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_q35_2b_environment_bootstrap_context_v1 import LEAK_LADDER
from datasets import Dataset
from export_prime_agent_role_sft_v1 import sha256_file
from procedural_harness_master_v1.taskset import (
    ProceduralHarnessMasterConfig,
    ProceduralHarnessMasterTaskset,
)

SCHEMA_VERSION = "qwen35-2b-environment-designer-corpus/v1"
BOOTSTRAP_SCHEMA_VERSION = "qwen35-2b-environment-bootstrap-context/v1"
SUMMARY_SCHEMA_VERSION = "qwen35-2b-interaction-curriculum-summary/v1"
SYSTEM_PROMPT = """You are the Environment Designer for a Prime Agents self-play curriculum.
Given one public task and an answer-free privileged interaction contract, write a training-only
environment scaffold that exposes the next valid control actions without revealing any exact task
answer or private evidence value. Preserve resource ownership, use the exact child name and path,
make the child report through agent_message.send to its parent, and tell the coordinator to yield
passively after spawning when no local work remains. Return only the scaffold text."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _message(role: str, content: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "reasoning_content": "",
        "tool_calls": [],
        "tool_call_id": "",
        "name": "",
    }


def _designer_contract(task: Any, leak_level: str) -> dict[str, Any]:
    oracle = task.data.oracle
    return {
        "episode_id": task.key,
        "public_system_prompt": task.data.system_prompt,
        "public_user_prompt": task.data.prompt,
        "generation_metadata": task.data.generation_metadata,
        "privileged_interaction_contract": {
            "expected_route": oracle.get("expected_route"),
            "resource_ownership": oracle.get("resource_ownership", {}),
            "children": [
                {
                    key: child[key]
                    for key in (
                        "name",
                        "resource_path",
                        "operation",
                        "message_contract",
                    )
                    if key in child
                }
                for child in oracle.get("children", [])
            ],
            "persistence_contract": {
                key: oracle["persistence_lease"][key]
                for key in ("path", "key")
                if key in oracle.get("persistence_lease", {})
            },
            "trajectory_contract": oracle["trajectory_contract"],
            "final_answer_keys": list(oracle["final_answer"]),
        },
        "leak_policy": {
            "stage": leak_level,
            "stage_index": LEAK_LADDER.index(leak_level),
            "exact_answer_allowed": False,
            "private_evidence_value_allowed": False,
            "next_valid_actions_allowed": True,
        },
    }


def export(
    *,
    bootstrap_path: Path,
    summary_path: Path,
    output_dir: Path,
    phase: str,
    student_snapshot: Path,
    student_revision: str,
    student_weight_sha: str,
    selection_count: int = 4,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite designer corpus: {output_dir}")
    if not 1 <= selection_count <= 4:
        raise ValueError("designer selection must be between one training row and four promotion rows")
    weight = student_snapshot / "model.safetensors"
    if (
        not student_snapshot.is_absolute()
        or not (student_snapshot / "STABLE").is_file()
        or sha256_file(weight) != student_weight_sha
    ):
        raise ValueError("designer student snapshot does not match its SHA-256")

    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if bootstrap.get("schema_version") != BOOTSTRAP_SCHEMA_VERSION:
        raise ValueError("unsupported environment bootstrap schema")
    leak_level = bootstrap.get("leak_level")
    if (
        bootstrap.get("gradient_updates") != 0
        or bootstrap.get("heldout_allowed") is not False
        or leak_level not in LEAK_LADDER
        or bootstrap.get("leak_stage_index") != LEAK_LADDER.index(leak_level)
        or bootstrap.get("leak_ladder") != list(LEAK_LADDER)
        or any(record.get("final_answer_in_context") is not False for record in bootstrap.get("records", []))
    ):
        raise ValueError("designer bootstrap must be answer-free train-gen action scaffolding")
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION or summary.get("phase") != phase:
        raise ValueError("designer source summary does not match its phase")
    gate = summary.get("gate") or {}
    qualifying = summary.get("qualifying") or []
    if (
        gate.get("acceptance_floor_relaxed") is not False
        or len(qualifying) < selection_count
        or summary.get("distinct_qualifying_task_keys", 0) < selection_count
    ):
        raise ValueError("designer source lacks the unchanged admission floor")

    selected_keys = sorted({item["task_key"] for item in qualifying})[:selection_count]
    if len(selected_keys) != selection_count:
        raise ValueError("designer source contains duplicate qualifying task keys")
    contexts = bootstrap.get("contexts") or {}
    records = {record["episode_id"]: record for record in bootstrap.get("records", [])}
    axes = bootstrap.get("axes") or []
    tasks_per_axis = bootstrap.get("tasks_per_axis")
    generated = {}
    task_axes = {}
    for axis in axes:
        tasks = ProceduralHarnessMasterTaskset(
            ProceduralHarnessMasterConfig(
                split="train_gen",
                count=tasks_per_axis,
                start_index=axis["start_index"],
                master_seed=bootstrap["master_seed"],
                curriculum_rung=axis["name"],
                private_payload_mode="finding_card",
            )
        ).load()
        generated.update({task.key: task for task in tasks})
        task_axes.update({task.key: axis["name"] for task in tasks})

    rows = []
    context_hashes = {}
    for task_key in selected_keys:
        task = generated.get(task_key)
        context = contexts.get(task_key)
        record = records.get(task_key)
        if task is None or not isinstance(context, str) or not isinstance(record, dict):
            raise ValueError(f"designer bootstrap lacks qualifying task {task_key}")
        context_sha = _sha256_text(context)
        if record.get("context_sha256") != context_sha:
            raise ValueError(f"designer context hash mismatch for {task_key}")
        prompt = _canonical_json(_designer_contract(task, leak_level))
        rows.append(
            {
                "messages": [
                    _message("system", SYSTEM_PROMPT),
                    _message("user", prompt),
                    _message("assistant", context),
                ],
                "tools": "[]",
                "axis": task_axes[task_key],
                "phase": phase,
                "task_key": task_key,
                "trace_id": f"environment-designer:{task_key}:{context_sha[:16]}",
                "role": "coordinator",
                "objective": "environment_designer",
            }
        )
        context_hashes[task_key] = context_sha

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "role": "coordinator",
        "objective": "environment_designer",
        "training_stage": "supervised_leak_fade_bootstrap",
        "leak_level": leak_level,
        "leak_stage_index": LEAK_LADDER.index(leak_level),
        "leak_ladder": list(LEAK_LADDER),
        "rows": len(rows),
        "selection_count": selection_count,
        "accepted_task_keys": selected_keys,
        "acceptance_floor_relaxed": False,
        "exact_answer_rows": 0,
        "student": {
            "snapshot": str(student_snapshot),
            "revision": student_revision,
            "weight_sha256": student_weight_sha,
            "dense_weight_mutated": True,
        },
        "source": {
            "bootstrap_path": str(bootstrap_path.resolve()),
            "bootstrap_sha256": sha256_file(bootstrap_path),
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": sha256_file(summary_path),
            "context_sha256": context_hashes,
        },
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--student-snapshot", type=Path, required=True)
    parser.add_argument("--student-revision", required=True)
    parser.add_argument("--student-weight-sha", required=True)
    parser.add_argument("--selection-count", type=int, default=4)
    args = parser.parse_args()
    manifest = export(
        bootstrap_path=args.bootstrap.resolve(),
        summary_path=args.summary.resolve(),
        output_dir=args.output_dir.resolve(),
        phase=args.phase,
        student_snapshot=args.student_snapshot.resolve(),
        student_revision=args.student_revision,
        student_weight_sha=args.student_weight_sha,
        selection_count=args.selection_count,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
