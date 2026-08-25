#!/usr/bin/env python3
"""Build a bounded, deduplicated replay corpus for one dense role model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import q35_2b_spade_dual_dense_loop_v1 as controller
from build_q35_2b_environment_bootstrap_context_v1 import LEAK_LADDER
from datasets import Dataset

SCHEMA_VERSION = "qwen35-2b-dual-role-replay/v2"
SUPPORTED_REPLAY_SCHEMA_VERSIONS = {
    "qwen35-2b-dual-role-replay/v1",
    SCHEMA_VERSION,
}
SOURCE_SCHEMA_VERSION = "qwen35-2b-interaction-joint-corpus/v2"
POSITIVE_PREFIX_SOURCE_SCHEMA_VERSION = "qwen35-2b-positive-prefix-corpus/v1"
DESIGNER_SOURCE_SCHEMA_VERSION = "qwen35-2b-environment-designer-corpus/v1"
REWARDED_DESIGNER_SOURCE_SCHEMA_VERSION = "qwen35-2b-spade-rewarded-designer-corpus/v1"
REPAIRED_DESIGNER_SOURCE_SCHEMA_VERSION = "qwen35-2b-spade-designer-repair-corpus/v1"
TRACK_FOR_ROLE = {"coordinator": "yield", "child": "child"}
REPLAY_PHASE_NAMESPACES = {"spade", "spade-repair"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _source_rows(path: Path, *, role: str, replay: bool) -> list[dict[str, Any]]:
    manifest = _json(path / "MANIFEST.json")
    parquet = path / "train.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(f"source corpus lacks train.parquet: {path}")
    if replay:
        if manifest.get("schema_version") not in SUPPORTED_REPLAY_SCHEMA_VERSIONS or manifest.get("role") != role:
            raise ValueError("prior replay schema or role mismatch")
    else:
        source_rows = manifest.get("rows_by_role", {}).get(role)
        complete_source = manifest.get("schema_version") == SOURCE_SCHEMA_VERSION
        positive_prefix_source = (
            manifest.get("schema_version") == POSITIVE_PREFIX_SOURCE_SCHEMA_VERSION
            and manifest.get("training_stage") == "hard_safety_validated_positive_prefix"
            and manifest.get("hard_safety_validated") is True
        )
        if (
            not (complete_source or positive_prefix_source)
            or manifest.get("selected_roles") != [role]
            or not isinstance(source_rows, int)
            or not 1 <= source_rows <= 4
            or manifest.get("rows_by_role") != {role: source_rows}
            or manifest.get("student", {}).get("dense_weight_mutated") is not True
        ):
            raise ValueError("new source is not one to four admitted rows for the mutated dense role")
    if sha256_file(parquet) != manifest.get("dataset", {}).get("sha256"):
        raise ValueError("source corpus parquet SHA-256 mismatch")
    rows = list(Dataset.from_parquet(str(parquet)))
    if any(row.get("role") != role or not isinstance(row.get("trace_id"), str) for row in rows):
        raise ValueError("source corpus contains an invalid role row")
    return rows


def _designer_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _json(path / "MANIFEST.json")
    parquet = path / "train.parquet"
    schema = manifest.get("schema_version")
    common_invalid = (
        manifest.get("role") != "coordinator"
        or manifest.get("objective") != "environment_designer"
        or not isinstance(manifest.get("rows"), int)
        or not 1 <= manifest["rows"] <= 4
        or manifest.get("selection_count") != manifest["rows"]
        or manifest.get("exact_answer_rows") != 0
    )
    static_valid = (
        schema == DESIGNER_SOURCE_SCHEMA_VERSION
        and manifest.get("acceptance_floor_relaxed") is False
        and manifest.get("leak_ladder") == list(LEAK_LADDER)
        and manifest.get("leak_level") in LEAK_LADDER
        and manifest.get("leak_stage_index") == LEAK_LADDER.index(manifest.get("leak_level"))
    )
    rewarded_valid = (
        schema == REWARDED_DESIGNER_SOURCE_SCHEMA_VERSION
        and manifest.get("training_stage") == "delayed_reward_filtered_coevolution"
        and isinstance(manifest.get("batch_id"), str)
        and 1 <= manifest.get("rows", 0) <= 2
    )
    repaired_valid = (
        schema == REPAIRED_DESIGNER_SOURCE_SCHEMA_VERSION
        and manifest.get("training_stage") == "scaffolded_schema_and_safety_repair"
        and isinstance(manifest.get("batch_id"), str)
        and manifest.get("hard_safety_validated") is True
        and 1 <= manifest.get("rows", 0) <= 2
    )
    if common_invalid or not (static_valid or rewarded_valid or repaired_valid):
        raise ValueError("designer source is not an admitted answer-free source")
    if not parquet.is_file() or sha256_file(parquet) != manifest.get("dataset", {}).get("sha256"):
        raise ValueError("designer source parquet SHA-256 mismatch")
    rows = list(Dataset.from_parquet(str(parquet)))
    if any(
        row.get("role") != "coordinator"
        or row.get("objective") != "environment_designer"
        or not isinstance(row.get("trace_id"), str)
        for row in rows
    ):
        raise ValueError("designer source contains an invalid row")
    return rows, manifest


def _replay_phase_rank(track: str, phase: Any) -> tuple[float, int]:
    if not isinstance(phase, str) or not phase:
        raise ValueError("replay row has an invalid phase")
    phase_track = track
    if ":" in phase:
        parts = phase.split(":", 2)
        if len(parts) != 3 or parts[0] not in REPLAY_PHASE_NAMESPACES:
            raise ValueError(f"replay phase {phase!r} has an invalid namespace")
        if parts[1] not in controller.PHASE_LADDERS or not parts[2]:
            raise ValueError(f"replay phase {phase!r} has an invalid embedded track")
        phase_track = parts[1]
        phase = parts[2]
    index = controller._phase_index(phase_track, phase)
    ladder = controller.PHASE_LADDERS[phase_track]
    relative_hardness = index / max(1, len(ladder) - 1)
    return relative_hardness, int(phase_track == track)


def _prior_anchors(
    rows: list[dict[str, Any]],
    *,
    role: str,
    excluded_trace_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit == 0:
        return []
    track = TRACK_FOR_ROLE[role]
    by_trace: dict[str, tuple[int, dict[str, Any]]] = {}
    for position, row in enumerate(rows):
        trace_id = row["trace_id"]
        if trace_id in excluded_trace_ids:
            continue
        by_trace.pop(trace_id, None)
        by_trace[trace_id] = (position, row)
    ranked = sorted(
        by_trace.values(),
        key=lambda item: (*_replay_phase_rank(track, item[1].get("phase")), item[0]),
        reverse=True,
    )[:limit]
    return [row for _, row in sorted(ranked)]


def combine(
    *,
    new_source: Path,
    output_dir: Path,
    role: str,
    prior_replay: Path | None = None,
    designer_source: Path | None = None,
    auxiliary_sources: list[Path] | None = None,
    max_rows: int = 64,
) -> dict[str, Any]:
    if role not in {"coordinator", "child"}:
        raise ValueError("role must be coordinator or child")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if designer_source is not None and role != "coordinator":
        raise ValueError("environment designer rows belong only to the coordinator model")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite replay corpus: {output_dir}")
    new_rows = _source_rows(new_source, role=role, replay=False)
    if not 1 <= len(new_rows) <= 4:
        raise ValueError("new source must contain one to four admitted rows")
    old_rows = _source_rows(prior_replay, role=role, replay=True) if prior_replay is not None else []
    auxiliary_sources = auxiliary_sources or []
    auxiliary_rows = [
        row
        for source in auxiliary_sources
        for row in _source_rows(source, role=role, replay=False)
    ]
    if len({row["trace_id"] for row in auxiliary_rows}) != len(auxiliary_rows):
        raise ValueError("auxiliary sources contain duplicate trajectories")
    designer_rows: list[dict[str, Any]] = []
    designer_manifest = None
    if designer_source is not None:
        designer_rows, designer_manifest = _designer_rows(designer_source)
    if (
        designer_rows
        and designer_manifest["schema_version"] == DESIGNER_SOURCE_SCHEMA_VERSION
        and len(designer_rows) != len(new_rows)
    ):
        raise ValueError("designer and coordinator interaction sources must use the same admission count")
    designer_metadata = None
    if designer_source is not None:
        if designer_manifest["schema_version"] == DESIGNER_SOURCE_SCHEMA_VERSION:
            trained_stage = designer_manifest["leak_stage_index"]
            designer_metadata = {
                "mode": "static_leak_ladder",
                "trained_leak_level": designer_manifest["leak_level"],
                "trained_stage_index": trained_stage,
                "next_stage_index": min(trained_stage + 1, len(LEAK_LADDER) - 1),
                "next_leak_level": LEAK_LADDER[min(trained_stage + 1, len(LEAK_LADDER) - 1)],
                "ladder": list(LEAK_LADDER),
                "promotion_step_size": 1,
            }
        else:
            designer_metadata = {
                "mode": (
                    "paired_hint_regret"
                    if designer_manifest["schema_version"] == REWARDED_DESIGNER_SOURCE_SCHEMA_VERSION
                    else "scaffolded_repair"
                ),
                "trained_batch_ids": [designer_manifest["batch_id"]],
                "selected_environment_ids": designer_manifest.get("selected_environment_ids", []),
                "training_stage": designer_manifest["training_stage"],
            }
    required_new = len(new_rows) + len(auxiliary_rows) + len(designer_rows)
    if max_rows < required_new:
        raise ValueError("replay cap cannot exclude newly admitted interaction or designer rows")
    new_trace_ids = {row["trace_id"] for row in [*new_rows, *auxiliary_rows, *designer_rows]}
    if len(new_trace_ids) != required_new:
        raise ValueError("new interaction, auxiliary, and designer sources overlap")
    prior = _prior_anchors(
        old_rows,
        role=role,
        excluded_trace_ids=new_trace_ids,
        limit=max_rows - required_new,
    )
    selected = [*prior, *new_rows, *auxiliary_rows, *designer_rows]
    if not new_trace_ids <= {row["trace_id"] for row in selected}:
        raise ValueError("replay cap excluded a newly admitted trajectory or design")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        parquet = temporary / "train.parquet"
        Dataset.from_list(selected).to_parquet(str(parquet))
        sources = {
            "new": {
                "path": str(new_source.resolve()),
                "manifest_sha256": sha256_file(new_source / "MANIFEST.json"),
                "train_parquet_sha256": sha256_file(new_source / "train.parquet"),
            },
            "prior": None,
            "auxiliary": [],
        }
        sources["auxiliary"] = [
            {
                "path": str(source.resolve()),
                "manifest_sha256": sha256_file(source / "MANIFEST.json"),
                "train_parquet_sha256": sha256_file(source / "train.parquet"),
            }
            for source in auxiliary_sources
        ]
        if prior_replay is not None:
            sources["prior"] = {
                "path": str(prior_replay.resolve()),
                "manifest_sha256": sha256_file(prior_replay / "MANIFEST.json"),
                "train_parquet_sha256": sha256_file(prior_replay / "train.parquet"),
            }
        sources["designer"] = None
        if designer_source is not None:
            sources["designer"] = {
                "path": str(designer_source.resolve()),
                "manifest_sha256": sha256_file(designer_source / "MANIFEST.json"),
                "train_parquet_sha256": sha256_file(designer_source / "train.parquet"),
            }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "role": role,
            "rows": len(selected),
            "new_rows": len(new_rows),
            "added_rows": required_new,
            "new_interaction_rows": len(new_rows),
            "new_auxiliary_rows": len(auxiliary_rows),
            "new_designer_rows": len(designer_rows),
            "new_partial_rows": sum(
                row.get("objective") == "interaction_positive_prefix"
                for row in [*new_rows, *auxiliary_rows]
            ),
            "max_rows": max_rows,
            "prior_selection": "hardest_phase_then_recency",
            "phase_counts": {
                phase: sum(row["phase"] == phase for row in selected)
                for phase in dict.fromkeys(row["phase"] for row in selected)
            },
            "objective_counts": {
                objective: sum(row.get("objective", "interaction") == objective for row in selected)
                for objective in dict.fromkeys(row.get("objective", "interaction") for row in selected)
            },
            "environment_designer": designer_metadata,
            "trace_ids": [row["trace_id"] for row in selected],
            "sources": sources,
            "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
        }
        (temporary / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-source", type=Path, required=True)
    parser.add_argument("--prior-replay", type=Path)
    parser.add_argument("--designer-source", type=Path)
    parser.add_argument("--auxiliary-source", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("coordinator", "child"), required=True)
    parser.add_argument("--max-rows", type=int, default=64)
    args = parser.parse_args()
    manifest = combine(
        new_source=args.new_source.resolve(),
        prior_replay=args.prior_replay.resolve() if args.prior_replay else None,
        designer_source=args.designer_source.resolve() if args.designer_source else None,
        auxiliary_sources=[path.resolve() for path in args.auxiliary_source],
        output_dir=args.output_dir.resolve(),
        role=args.role,
        max_rows=args.max_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
