#!/usr/bin/env python3
"""Materialize the fresh B-IPC1 train, validation, and heldout banks."""

from __future__ import annotations

import argparse
import json
import shutil
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

from prime_rl.phase_b_contract import file_sha256
from prime_rl.phase_b_ipc1 import (
    ACTIONS,
    SELECTIONS,
    canonical_bank_sha256,
    select_balanced_rows,
    validate_bank_disjointness,
)

SCHEMA_VERSION = "q35-2b-phase-b-ipc1-bank/v1"
TASKSET_COMMIT = "5283a85a01b5e8a065b3d2db17f9efa6aa0f3b2f"
TASKSET_SHA256 = "15332134b09d5b6bccfaaf03166ddb5aa88b8bafe84a01e34b7409b218f50499"
TRAINING_PARQUET_SHA256 = "256328cbdb33dc8880370688e2c6f4256341f644757a9d9d033bb5febc74e12e"
ROOT_SYSTEM_MESSAGE_SHA256 = "59514a723921737f4ad6fcab55c82aee464050297bf343591ea8e2d7950c60b0"
FAMILIES = ("document_adaptive_d0", "document_adaptive_d1", "document_adaptive_d2")
SOURCE_IDENTITIES = (
    ("e33_c54_v1", "81fc8f03f3a7346bd62f542fdce14dd7619127502488b1085c6ce396dc7c4549"),
    ("e33_action_v2", "7f37c0dfbf2d500e0f64b6a4e65f33eaa6efa884b8555fba31e8e9e7e3f13fc8"),
    ("e33_nonroot_v3", "52b50f8f57d2ad6cf084c13bfe429293b304824c76df2c979b6e7793d1344575"),
    ("b1_manifest", "a1b9fa94dc65640cf5bf174edf6cf6581bb1e50e32f454cf474c9230e116f10e"),
    ("b1_training_selection", "f049818029e3b241d0b3b777696e500d0f7320f9614f6a49cc57def13dfc9b1b"),
    ("b1_heldout_selection", "e086b751155c2c5e8d000cd60d3ff999bbdd91e6faaf3a3c68a5913de5957aca"),
    ("br5_selection", "8e160b9214aeb5cc971abf472cb31c0173bdfeee2d56fea98620dc87b166b3fe"),
    ("hic0_manifest", "9e0498d210bd87b29e0385a938ce1717f61e45d24c1dde32f49ad95307a051fb"),
    ("hic0_selection", "084dc6083a56a03f7fd4b51da350c8b5e916f37db5bb4a5baa1ef50945026618"),
    ("a0_mechanism_bank", "0e940fedadf3f11591e65f963c082c4b96025b883dac68255a07f4cb3f38b9a1"),
    ("a0_nocache_bank", "b77df46145d67e9147f42b9dd1e403a6253955e2a67bab3c95490357fe255ea3"),
    ("a1_train_bank", "cea92c57536ec7a93e68c64c4a01669e11dab1dc483a86c711c97c1181bb08d8"),
    ("a1_validation_bank", "8e87309df0c37cce7bbd0bb98a3daed098f0b354f0c00ae15c27024047539c5e"),
    ("a1_heldout_bank", "204e6605fca20f94b23468fbd9f56b86ca16e8296191dd09b877a021c9ab2381"),
    ("a0_nocache_disjointness", "88c730b9688abcbc7a2df4792706df2cb86962e5f1be8fd04ac75f6f5766644f"),
    ("a1_disjointness", "44c78fb0680d837a284c0d63b6154b51ea1b76ea4f3a0cf0dac7de56c96974e2"),
)


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _runtime_from_training(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    root_messages = [row["messages"][0] for row in rows if row.get("role_scope") == "root"]
    if len(root_messages) != 32 or len({canonical_bank_sha256(message) for message in root_messages}) != 1:
        raise ValueError("B-IPC1 source does not contain one exact root runtime across 32 rows")
    source = root_messages[0]
    if canonical_bank_sha256(source) != ROOT_SYSTEM_MESSAGE_SHA256:
        raise ValueError("B-IPC1 source root system message differs")
    prefix = f"{ROOT_COORDINATOR_CONTRACT}\n\n"
    content = source.get("content")
    if source.get("role") != "system" or not isinstance(content, str) or not content.startswith(prefix):
        raise ValueError("B-IPC1 root runtime boundary is invalid")
    runtime = {"role": "system", "content": content[len(prefix) :]}
    if {"role": "system", "content": prefix + runtime["content"]} != {
        "role": source["role"],
        "content": source["content"],
    }:
        raise ValueError("B-IPC1 runtime extraction does not round-trip")
    return runtime, source


def _candidate_pool(*, split: str, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    specification = SELECTIONS[split]
    taskset_split = "train" if split == "train" else "eval"
    taskset = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split=taskset_split,
            families=FAMILIES,
            instances_per_template=specification["instance_stop"] - specification["instance_start"],
            instance_offset=specification["instance_start"],
            seed=specification["seed"],
        )
    )
    rows: list[dict[str, Any]] = []
    for task in taskset.load():
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
    variants = sorted({int(row["task_key"].split("-v", 1)[1].split("-", 1)[0]) for row in rows})
    expected_variants = [0, 1, 2, 3] if split == "train" else [4, 5]
    if variants != expected_variants:
        raise ValueError(f"B-IPC1 {split} taskset variants differ: {variants}")
    return rows


def _source_records(source_paths: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, set[str]]]:
    if tuple(source_paths) != tuple(name for name, _sha in SOURCE_IDENTITIES):
        raise ValueError("B-IPC1 prior overlap source order differs")
    records: list[dict[str, Any]] = []
    key_sets: dict[str, set[str]] = {}
    hash_sets: dict[str, set[str]] = {}
    for name, expected_sha in SOURCE_IDENTITIES:
        path = source_paths[name]
        observed_sha = file_sha256(path)
        if observed_sha != expected_sha:
            raise ValueError(f"B-IPC1 overlap source hash differs: {name}")
        value = _load(path)
        keys: set[str] = set()
        row_hashes: set[str] = set()
        if name.startswith("e33_"):
            keys.update(value["task_keys"])
        elif name == "b1_training_selection":
            keys.update(value["task_keys"])
        elif name == "b1_heldout_selection":
            keys.update(value["task_keys"])
        elif name == "br5_selection":
            keys.update(value["probe_task_keys"])
        elif name == "hic0_selection":
            keys.update(value["task_keys"])
        elif name in {"b1_manifest", "hic0_manifest"}:
            row_hashes.update(value["row_canonical_sha256"])
        elif name.startswith("a0_") and name.endswith("bank"):
            for example in value["examples"]:
                keys.add(example["example_id"])
                row_hashes.add(canonical_bank_sha256(example))
        elif name.startswith("a1_") and name.endswith("bank"):
            for row in value["bank"]["records"]:
                keys.add(row["evidence_id"])
                keys.update(query["query_id"] for query in row["queries"])
                row_hashes.add(canonical_bank_sha256(row))
        records.append(
            {
                "name": name,
                "source_commit": "a8f347c9a5fdf1c2d532c6527ce169cff0000a07" if name.startswith("a") else None,
                "source_path": str(path),
                "source_file_sha256": observed_sha,
                "task_keys": sorted(keys),
                "canonical_row_sha256": sorted(row_hashes),
            }
        )
        key_sets[name] = keys
        hash_sets[name] = row_hashes
    return records, key_sets, hash_sets


def build(
    *, training_parquet: Path, taskset_source: Path, source_paths: dict[str, Path]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if file_sha256(training_parquet) != TRAINING_PARQUET_SHA256:
        raise ValueError("B-IPC1 runtime source parquet differs")
    if file_sha256(taskset_source) != TASKSET_SHA256:
        raise ValueError("B-IPC1 taskset source differs")
    source_rows = pq.read_table(training_parquet).to_pylist()
    runtime, source_message = _runtime_from_training(source_rows)
    selected: dict[str, list[dict[str, Any]]] = {}
    selections: dict[str, Any] = {}
    pool_evidence: list[dict[str, Any]] = []
    for split in ("train", "validation", "heldout"):
        pool = _candidate_pool(split=split, runtime=runtime)
        selected[split], selections[split] = select_balanced_rows(pool, split=split)
        pool_evidence.append(
            {
                "split": split,
                "taskset_split": "train" if split == "train" else "eval",
                "candidate_rows": len(pool),
                "candidate_row_list_sha256": canonical_bank_sha256(pool),
                "candidate_task_keys_sha256": canonical_bank_sha256([row["task_key"] for row in pool]),
            }
        )
    prior_records, prior_keys, prior_hashes = _source_records(source_paths)
    overlap = validate_bank_disjointness(
        selected,
        excluded_key_sets=prior_keys,
        excluded_row_hash_sets=prior_hashes,
    )
    closure = {
        "schema_version": "q35-2b-b-ipc1-prior-overlap-closure/v1",
        "source_records": prior_records,
        "source_record_order_sha256": canonical_bank_sha256(prior_records),
        "union_task_keys": sorted(set().union(*prior_keys.values())),
        "union_canonical_row_sha256": sorted(set().union(*prior_hashes.values())),
        "overlap_evidence": overlap,
    }
    runtime_evidence = {
        "schema_version": "q35-2b-b-ipc1-runtime-source/v1",
        "training_root_rows": 32,
        "unique_training_root_system_messages": 1,
        "root_system_message_sha256": canonical_bank_sha256(source_message),
        "runtime_message": runtime,
        "runtime_message_sha256": canonical_bank_sha256(runtime),
        "reconstruction_byte_exact": True,
    }
    generation = {
        "taskset_commit": TASKSET_COMMIT,
        "taskset_source_sha256": TASKSET_SHA256,
        "families": list(FAMILIES),
        "action_order": list(ACTIONS),
        "d0_suffix": ":solve-anchor-1",
        "pool_evidence": pool_evidence,
    }
    return selected, selections, closure, {"runtime": runtime_evidence, "generation": generation}


def export(
    *, training_parquet: Path, taskset_source: Path, source_paths: dict[str, Path], output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite B-IPC1 bank: {output_dir}")
    selected, selections, closure, evidence = build(
        training_parquet=training_parquet,
        taskset_source=taskset_source,
        source_paths=source_paths,
    )
    output_dir.mkdir(parents=True)
    for split, rows in selected.items():
        pq.write_table(pa.Table.from_pylist(rows), output_dir / f"{split}.parquet")
        normalized = pq.read_table(output_dir / f"{split}.parquet").to_pylist()
        if [(row["task_key"], row["action"]) for row in normalized] != [
            (row["task_key"], row["action"]) for row in rows
        ]:
            raise ValueError(f"B-IPC1 {split} parquet changed selected key/action order")
        selected[split] = normalized
        selections[split]["row_list_canonical_sha256"] = canonical_bank_sha256(normalized)
        selections[split]["row_canonical_sha256"] = [canonical_bank_sha256(row) for row in normalized]
    prior_keys = {
        record["name"]: set(record["task_keys"])
        for record in closure["source_records"]
    }
    prior_hashes = {
        record["name"]: set(record["canonical_row_sha256"])
        for record in closure["source_records"]
    }
    closure["overlap_evidence"] = validate_bank_disjointness(
        selected,
        excluded_key_sets=prior_keys,
        excluded_row_hash_sets=prior_hashes,
    )
    json_values = {
        "runtime-source.json": evidence["runtime"],
        "prior-overlap-closure-v1.json": closure,
        **{f"{split}-selection.json": selection for split, selection in selections.items()},
    }
    for name, value in json_values.items():
        (output_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copyfile(taskset_source, output_dir / "taskset-5283a85.py")
    artifacts = {
        path.name: file_sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "rows": [{"split": split, "count": len(selected[split])} for split in selected],
        "generation": evidence["generation"],
        "selection_hashes": [
            {
                "split": split,
                "ordered_task_key_sha256": selections[split]["ordered_task_key_sha256"],
                "ordered_key_action_sha256": selections[split]["ordered_key_action_sha256"],
                "row_list_canonical_sha256": selections[split]["row_list_canonical_sha256"],
            }
            for split in selected
        ],
        "freshness": closure["overlap_evidence"],
        "artifacts": artifacts,
        "generator": {
            "source_path": "scripts/export_phase_b_ipc1_bank_v1.py",
            "source_sha256": file_sha256(Path(__file__)),
        },
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-parquet", type=Path, required=True)
    parser.add_argument("--taskset-source", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources: dict[str, Path] = {}
    for item in args.source:
        name, separator, value = item.partition("=")
        if not separator or name in sources:
            raise ValueError(f"invalid or duplicate B-IPC1 source binding: {item}")
        sources[name] = Path(value).resolve()
    export(
        training_parquet=args.training_parquet.resolve(),
        taskset_source=args.taskset_source.resolve(),
        source_paths=sources,
        output_dir=args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
