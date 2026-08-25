"""Export an admitted Prime Agent teacher bank into one role-specific SFT dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

Role = Literal["orchestrator", "child"]
SCHEMA_VERSION = "q38-to-q35-2b-teacher-admission/v1"
TRUNCATED_STOPS = {
    "max_turns",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "context_length",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(record: dict[str, Any] | None) -> float | None:
    if not isinstance(record, dict):
        return None
    value = record.get("score", record.get("value"))
    return float(value) if isinstance(value, int | float) else None


def _text_only_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    if all(isinstance(part, dict) and part.get("type") == "text" for part in content):
        return "".join(str(part.get("text", "")) for part in content)
    raise ValueError("role SFT export supports text-only Prime Agent messages")


def _wire_message(message: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in message.items() if value is not None}
    if "content" in result:
        result["content"] = _text_only_content(result["content"])
    return result


def _branch_paths(trace: dict[str, Any]) -> list[list[int]]:
    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"trace {trace.get('id')} has no message graph")
    parents: list[int | None] = []
    parent_nodes: set[int] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"trace {trace.get('id')} node {index} is not an object")
        parent = node.get("parent")
        if parent is not None and (not isinstance(parent, int) or not 0 <= parent < len(nodes)):
            raise ValueError(f"trace {trace.get('id')} node {index} has invalid parent {parent}")
        parents.append(parent)
        if parent is not None:
            parent_nodes.add(parent)

    paths: list[list[int]] = []
    for leaf in (index for index in range(len(nodes)) if index not in parent_nodes):
        path: list[int] = []
        seen: set[int] = set()
        node_index: int | None = leaf
        while node_index is not None:
            if node_index in seen:
                raise ValueError(f"trace {trace.get('id')} contains a graph cycle")
            seen.add(node_index)
            path.append(node_index)
            node_index = parents[node_index]
        paths.append(list(reversed(path)))
    return paths


def role_rows(
    trace: dict[str, Any],
    *,
    episode_id: str,
    axis: str,
    role: Role,
) -> tuple[list[dict[str, Any]], list[str]]:
    nodes = trace["nodes"]
    roots = [index for index, node in enumerate(nodes) if node.get("parent") is None]
    if not roots:
        raise ValueError(f"trace {trace.get('id')} has no graph root")
    primary_root = roots[0]

    lineage: dict[int, set[str | None]] = {}
    for call in trace.get("calls", []):
        node_index = call.get("node")
        if node_index is None:
            continue
        if not isinstance(node_index, int) or not 0 <= node_index < len(nodes):
            raise ValueError(f"trace {trace.get('id')} has a call with invalid node {node_index}")
        lineage.setdefault(node_index, set()).add(call.get("client_session_id"))

    selected_paths = [path for path in _branch_paths(trace) if (path[0] == primary_root) == (role == "orchestrator")]
    if not selected_paths:
        if role == "child" and axis == "natural_direct_control":
            return [], []
        raise ValueError(f"trace {trace.get('id')} has no {role} branch")

    trained_nodes: set[int] = set()
    rows: list[dict[str, Any]] = []
    sessions: list[str] = []
    tools = json.dumps(trace.get("tools") or [], sort_keys=True, separators=(",", ":"))
    for branch_index, path in enumerate(selected_paths):
        sampled_nodes = [index for index in path if nodes[index].get("sampled") is True]
        if not sampled_nodes:
            raise ValueError(f"trace {trace.get('id')} {role} branch has no sampled messages")
        duplicates = trained_nodes.intersection(sampled_nodes)
        if duplicates:
            raise ValueError(f"trace {trace.get('id')} {role} branches share sampled nodes {sorted(duplicates)}")
        trained_nodes.update(sampled_nodes)

        branch_sessions: set[str] = set()
        for node_index in sampled_nodes:
            candidates = lineage.get(node_index, set())
            if len(candidates) != 1 or None in candidates:
                raise ValueError(f"trace {trace.get('id')} sampled node {node_index} has ambiguous session lineage")
            session = next(iter(candidates))
            assert session is not None
            branch_sessions.add(session)
        if len(branch_sessions) != 1:
            raise ValueError(f"trace {trace.get('id')} {role} branch crosses {len(branch_sessions)} sessions")
        session = next(iter(branch_sessions))
        sessions.append(session)
        rows.append(
            {
                "messages": [_wire_message(nodes[index]["message"]) for index in path],
                "tools": tools,
                "axis": axis,
                "episode_id": episode_id,
                "trace_id": trace["id"],
                "role": role,
                "branch_index": branch_index,
            }
        )
    return rows, sessions


def validate_admitted_trace(trace: dict[str, Any], axis: str) -> None:
    trace_id = trace.get("id")
    if not trace.get("ok") or trace.get("errors"):
        raise ValueError(f"admitted trace {trace_id} is errored")
    if trace.get("stop_condition") in TRUNCATED_STOPS:
        raise ValueError(f"admitted trace {trace_id} is truncated")
    calls = trace.get("calls") or []
    if calls and calls[-1].get("finish_reason") == "length":
        raise ValueError(f"admitted trace {trace_id} ended at provider length")
    metrics = trace.get("metrics") or {}
    if float(metrics.get("final_answer_exact", 0.0)) != 1.0:
        raise ValueError(f"admitted trace {trace_id} lacks exact final output")
    if axis != "natural_direct_control":
        reward = _score((trace.get("rewards") or {}).get("harness_score"))
        if reward != 1.0:
            raise ValueError(f"admitted asynchronous trace {trace_id} is not a hard success")


def load_admitted_rows(
    manifest: dict[str, Any], source_root: Path, role: Role
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported admission manifest schema: {manifest.get('schema_version')}")
    teacher_model = manifest.get("teacher", {}).get("model")
    required_axes = manifest.get("required_axes")
    if not isinstance(required_axes, dict) or not required_axes:
        raise ValueError("admission manifest must declare required_axes")

    rows: list[dict[str, Any]] = []
    admitted_by_axis: Counter[str] = Counter()
    branch_count_by_axis: Counter[str] = Counter()
    source_audit: list[dict[str, Any]] = []
    seen_trace_ids: set[str] = set()
    seen_episode_ids: set[str] = set()

    for source in manifest.get("sources", []):
        axis = source.get("axis")
        if axis not in required_axes:
            raise ValueError(f"source declares unexpected axis {axis}")
        trace_path = source_root / source["trace_path"]
        actual_hash = sha256_file(trace_path)
        if actual_hash != source.get("sha256"):
            raise ValueError(f"source hash mismatch for {trace_path}: {actual_hash}")
        versions_path = source_root / source["versions_path"]
        actual_versions_hash = sha256_file(versions_path)
        if actual_versions_hash != source.get("versions_sha256"):
            raise ValueError(f"source provenance hash mismatch for {versions_path}: {actual_versions_hash}")
        accepted_ids = source.get("accepted_trace_ids")
        if not isinstance(accepted_ids, list) or len(set(accepted_ids)) != len(accepted_ids):
            raise ValueError(f"source {axis} must list unique accepted_trace_ids")
        accepted = set(accepted_ids)
        found: set[str] = set()

        with trace_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                episode = json.loads(line)
                episode_id = episode.get("id")
                traces = episode.get("traces")
                if not isinstance(traces, list):
                    raise ValueError(f"{trace_path}:{line_number} is not a wire episode")
                for trace in traces:
                    trace_id = trace.get("id")
                    if trace_id not in accepted:
                        continue
                    if trace_id in seen_trace_ids or episode_id in seen_episode_ids:
                        raise ValueError(f"duplicate admitted trace or episode: {trace_id}/{episode_id}")
                    if trace.get("agent", {}).get("config", {}).get("model") != teacher_model:
                        raise ValueError(f"trace {trace_id} was not sampled from {teacher_model}")
                    validate_admitted_trace(trace, axis)
                    trace_rows, _ = role_rows(
                        trace,
                        episode_id=episode_id,
                        axis=axis,
                        role=role,
                    )
                    rows.extend(trace_rows)
                    found.add(trace_id)
                    seen_trace_ids.add(trace_id)
                    seen_episode_ids.add(episode_id)
                    admitted_by_axis[axis] += 1
                    branch_count_by_axis[axis] += len(trace_rows)
        missing = accepted - found
        if missing:
            raise ValueError(f"source {axis} is missing admitted trace IDs: {sorted(missing)}")
        source_audit.append(
            {
                "axis": axis,
                "trace_path": source["trace_path"],
                "sha256": actual_hash,
                "versions_path": source["versions_path"],
                "versions_sha256": actual_versions_hash,
                "accepted_trace_ids": accepted_ids,
            }
        )

    for axis, minimum in required_axes.items():
        if admitted_by_axis[axis] < int(minimum):
            raise ValueError(f"axis {axis} has {admitted_by_axis[axis]} admitted trajectories; requires {minimum}")
    if role == "child" and branch_count_by_axis["natural_direct_control"]:
        raise ValueError("direct-control traces unexpectedly produced child branches")

    audit = {
        "schema_version": "q38-to-q35-2b-role-corpus/v1",
        "role": role,
        "teacher": manifest["teacher"],
        "admitted_trajectories_by_axis": dict(sorted(admitted_by_axis.items())),
        "rows_by_axis": dict(sorted(branch_count_by_axis.items())),
        "row_count": len(rows),
        "sources": source_audit,
    }
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("admission_manifest", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--role", choices=("orchestrator", "child"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite role corpus: {args.output_dir}")
    manifest = json.loads(args.admission_manifest.read_text(encoding="utf-8"))
    rows, audit = load_admitted_rows(manifest, args.source_root, args.role)
    if not rows:
        raise SystemExit(f"no {args.role} rows were exported")

    audit["admission_manifest"] = {
        "path": str(args.admission_manifest.resolve()),
        "sha256": sha256_file(args.admission_manifest),
    }
    audit["source_root"] = str(args.source_root.resolve())

    from datasets import Dataset

    args.output_dir.mkdir(parents=True)
    parquet_path = args.output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet_path))
    audit["dataset"] = {
        "path": parquet_path.name,
        "sha256": sha256_file(parquet_path),
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"exported {audit['row_count']} {args.role} row(s) from "
        f"{sum(audit['admitted_trajectories_by_axis'].values())} admitted trajectories"
    )


if __name__ == "__main__":
    main()
