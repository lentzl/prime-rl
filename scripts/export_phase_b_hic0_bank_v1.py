#!/usr/bin/env python3
"""Export the prospective B-HIC0 fresh identity-carrier diagnostic bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from dual_policy_openai_proxy_v1 import ROOT_COORDINATOR_CONTRACT
from export_q35_2b_adaptive_cognition_sft_v1 import _row
from subagent_communication_v1.taskset import (
    ADAPTIVE_DOCUMENT_DEPTHS,
    SubagentCommunicationConfig,
    SubagentCommunicationTaskset,
)

SCHEMA_VERSION = "q35-2b-b-hic0-identity-carrier-bank/v1"
FAMILIES = ("document_adaptive_d0", "document_adaptive_d1", "document_adaptive_d2")
ACTIONS = ("solve_owned", "delegate_terminal", "delegate_coordinator")
GENERATOR_SEED_PAYLOAD = "q35-2b-b-h-identity-carrier-v1:bank"
GENERATOR_SEED_SHA256 = "87fa52cbc9b925761c786def9cd929bd54b9271b7aa9197215aa989b2d2f337d"
GENERATOR_SEED = 2281329355
INSTANCE_OFFSET = 35100
INSTANCES = 2
TASKSET_COMMIT = "5283a85a01b5e8a065b3d2db17f9efa6aa0f3b2f"
TASKSET_SHA256 = "15332134b09d5b6bccfaaf03166ddb5aa88b8bafe84a01e34b7409b218f50499"
ROOT_SYSTEM_MESSAGE_SHA256 = "59514a723921737f4ad6fcab55c82aee464050297bf343591ea8e2d7950c60b0"
EXPECTED_ROW_LIST_SHA256 = "b6fc31c9bdbce338e9df62c1f241aa57ce96cf2f0b8b8f454442c9064e274a5b"
EXPECTED_TRAIN_KEYS_SHA256 = "cd24743d41543e07485a6c9c690e3622847807f6ec813374f99cc5115236fb9a"
EXPECTED_TRAIN_BATCHES_SHA256 = "4bc83af062e1ca3f152daf32199ea6c1708c1e7f5dc54cad281ad67322170afb"
EXPECTED_HELDOUT_KEYS_SHA256 = "a331879c34cbc242e5af3775e520b5f6347a8a830e14cab52b95f108ea349d7d"
EXPECTED_HELDOUT_KEY_ACTION_SHA256 = "edf43c186f2c67f467fd60f309dfc0ea4fe34b8823159ce9d19a0736254c8f01"
EXPECTED_B1_HELDOUT_SELECTION_SHA256 = "e086b751155c2c5e8d000cd60d3ff999bbdd91e6faaf3a3c68a5913de5957aca"
EXPECTED_BR5_SELECTION_SHA256 = "8e160b9214aeb5cc971abf472cb31c0173bdfeee2d56fea98620dc87b166b3fe"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_from_training(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    root_messages = [row["messages"][0] for row in rows if row.get("role_scope") == "root"]
    if len(root_messages) != 32 or len({canonical_sha256(message) for message in root_messages}) != 1:
        raise ValueError("training source does not contain one exact root runtime message across 32 rows")
    source = root_messages[0]
    if canonical_sha256(source) != ROOT_SYSTEM_MESSAGE_SHA256:
        raise ValueError("training root system message differs from the prospective hash")
    expected_prefix = f"{ROOT_COORDINATOR_CONTRACT}\n\n"
    content = source.get("content")
    if source.get("role") != "system" or not isinstance(content, str) or not content.startswith(expected_prefix):
        raise ValueError("training root message does not preserve the exact coordinator/runtime boundary")
    runtime = {"role": "system", "content": content[len(expected_prefix) :]}
    reconstructed = {"role": "system", "content": expected_prefix + runtime["content"]}
    if reconstructed != {"role": source["role"], "content": source["content"]}:
        raise ValueError("runtime extraction does not reconstruct the exact root message")
    return runtime, source


def build(
    *, training_parquet: Path, taskset_source: Path, b1_heldout_selection: Path, br5_selection: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if file_sha256(training_parquet) != "256328cbdb33dc8880370688e2c6f4256341f644757a9d9d033bb5febc74e12e":
        raise ValueError("B-HIC0 provenance training parquet differs")
    if file_sha256(taskset_source) != TASKSET_SHA256:
        raise ValueError("B-HIC0 taskset source differs")
    training_rows = pq.read_table(training_parquet).to_pylist()
    training_keys = [row["task_key"] for row in training_rows]
    training_batches = [training_keys[index : index + 12] for index in range(0, 48, 12)]
    if canonical_sha256(training_keys) != EXPECTED_TRAIN_KEYS_SHA256:
        raise ValueError("B-HIC0 prior training key order differs")
    if canonical_sha256(training_batches) != EXPECTED_TRAIN_BATCHES_SHA256:
        raise ValueError("B-HIC0 prior training batch order differs")
    if file_sha256(b1_heldout_selection) != EXPECTED_B1_HELDOUT_SELECTION_SHA256:
        raise ValueError("B-HIC0 prior B1 heldout selection differs")
    if file_sha256(br5_selection) != EXPECTED_BR5_SELECTION_SHA256:
        raise ValueError("B-HIC0 prior BR5 selection differs")
    b1_heldout = json.loads(b1_heldout_selection.read_text(encoding="utf-8"))["task_keys"]
    br5 = json.loads(br5_selection.read_text(encoding="utf-8"))["probe_task_keys"]
    runtime, source_message = _runtime_from_training(training_rows)
    taskset = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="eval",
            families=FAMILIES,
            instances_per_template=INSTANCES,
            instance_offset=INSTANCE_OFFSET,
            seed=GENERATOR_SEED,
        )
    )
    tasks = taskset.load()
    rows: list[dict[str, Any]] = []
    for task in tasks:
        key = task.data.name
        if task.data.family == "document_adaptive_d0":
            key += ":solve-anchor-1"
        rows.append(
            _row(
                runtime=runtime,
                prompt=task.data.prompt,
                key=key,
                depth=ADAPTIVE_DOCUMENT_DEPTHS[task.data.family],
                role_scope="root",
                root=True,
            )
        )
    keys = [row["task_key"] for row in rows]
    key_actions = [{"task_key": row["task_key"], "expected_action": row["action"]} for row in rows]
    if len(rows) != 12 or canonical_sha256(keys) != EXPECTED_HELDOUT_KEYS_SHA256:
        raise ValueError("B-HIC0 diagnostic key order differs")
    if canonical_sha256(key_actions) != EXPECTED_HELDOUT_KEY_ACTION_SHA256:
        raise ValueError("B-HIC0 diagnostic key/action pairs differ")
    if canonical_sha256(rows) != EXPECTED_ROW_LIST_SHA256:
        raise ValueError("B-HIC0 rows differ from the independent cross-check")
    prior_keys = set(training_keys) | set(b1_heldout) | set(br5)
    if set(keys).intersection(prior_keys):
        raise ValueError("B-HIC0 diagnostic keys overlap a prior B bank")
    if Counter(row["action"] for row in rows) != Counter({action: 4 for action in ACTIONS}):
        raise ValueError("B-HIC0 rows are not action-balanced")
    runtime_evidence = {
        "schema_version": "q35-2b-b-hic0-runtime-source/v1",
        "training_root_rows": 32,
        "unique_training_root_system_messages": 1,
        "root_system_message_sha256": canonical_sha256(source_message),
        "runtime_message": runtime,
        "runtime_message_sha256": canonical_sha256(runtime),
        "reconstruction_byte_exact": True,
    }
    selection = {
            "schema_version": "q35-2b-b-hic0-identity-carrier-selection/v1",
            "task_keys": keys,
            "ordered_task_key_sha256": canonical_sha256(keys),
            "key_actions": key_actions,
            "ordered_key_action_sha256": canonical_sha256(key_actions),
            "expected_action_counts": dict(Counter(row["action"] for row in rows)),
    }
    freshness = {
        "zero_exact_key_overlap": True,
        "prior_key_union_count": len(prior_keys),
        "prior_sources": {
            "e33_and_b1_training": {
                "ordered_key_sha256": canonical_sha256(training_keys),
                "keys": len(training_keys),
            },
            "b1_heldout": {
                "selection_sha256": file_sha256(b1_heldout_selection),
                "keys": len(b1_heldout),
            },
            "br5": {"selection_sha256": file_sha256(br5_selection), "keys": len(br5)},
        },
        "diagnostic_rows_forbidden_future_training": True,
    }
    return rows, runtime_evidence, selection, freshness


def export(
    *,
    training_parquet: Path,
    taskset_source: Path,
    b1_heldout_selection: Path,
    br5_selection: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite B-HIC0 bank: {output_dir}")
    rows, runtime_evidence, selection, freshness = build(
        training_parquet=training_parquet,
        taskset_source=taskset_source,
        b1_heldout_selection=b1_heldout_selection,
        br5_selection=br5_selection,
    )
    output_dir.mkdir(parents=True)
    runtime_path = output_dir / "runtime-source.json"
    heldout_selection_path = output_dir / "selection.json"
    parquet_path = output_dir / "heldout.parquet"
    taskset_snapshot_path = output_dir / "taskset-5283a85.py"
    runtime_path.write_text(json.dumps(runtime_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    heldout_selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(taskset_source, taskset_snapshot_path)
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "rows": 12,
        "row_list_canonical_sha256": canonical_sha256(rows),
        "row_canonical_sha256": [canonical_sha256(row) for row in rows],
        "taskset": {
            "commit": TASKSET_COMMIT,
            "source_path": (
                "experiments/qwen35-2b-latent-coordinator-v1/phase-b-hic0-identity-carrier-bank-v1/"
                "taskset-5283a85.py"
            ),
            "source_sha256": file_sha256(taskset_source),
        },
        "generator": {
            "source_path": "scripts/export_phase_b_hic0_bank_v1.py",
            "source_sha256": file_sha256(Path(__file__)),
        },
        "generation": {
            "seed_payload": GENERATOR_SEED_PAYLOAD,
            "seed_sha256": GENERATOR_SEED_SHA256,
            "split": "eval",
            "families": list(FAMILIES),
            "template_variants": [4, 5],
            "instances_per_template": INSTANCES,
            "instance_offset": INSTANCE_OFFSET,
            "seed": GENERATOR_SEED,
            "order": "instance_then_variant_then_family",
            "d0_suffix": ":solve-anchor-1",
        },
        "freshness": freshness,
        "artifacts": {
            path.name: file_sha256(path)
            for path in (
                parquet_path,
                runtime_path,
                heldout_selection_path,
                taskset_snapshot_path,
            )
        },
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-parquet", type=Path, required=True)
    parser.add_argument("--taskset-source", type=Path, required=True)
    parser.add_argument("--b1-heldout-selection", type=Path, required=True)
    parser.add_argument("--br5-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = export(
        training_parquet=args.training_parquet.resolve(),
        taskset_source=args.taskset_source.resolve(),
        b1_heldout_selection=args.b1_heldout_selection.resolve(),
        br5_selection=args.br5_selection.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
