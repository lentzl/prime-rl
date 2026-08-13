#!/usr/bin/env python3
"""Build an auditable Prime Agent teacher bootstrap from verified 27B traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from datasets import Dataset
from verifiers.v1 import WireTrace
from verifiers.v1.cli.output import read_episodes

try:
    from scripts.export_sft import sft_row, sft_rows
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from export_sft import sft_row, sft_rows

Cohort = Literal["ownership", "communication"]
AdmissionQuality = Literal["clean", "conditioned_causal_near_clean"]
BASE_MODEL = "Qwen/Qwen3.5-27B"
BASE_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(trace, name: str) -> float | None:
    value = trace.metrics.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def parse_count_requirement(value: str) -> tuple[str, int]:
    try:
        name, minimum = value.rsplit("=", 1)
        parsed = int(minimum)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("count requirements use KEY=MIN") from exc
    if not name or parsed < 0:
        raise argparse.ArgumentTypeError("count requirements need a key and a non-negative integer")
    return name, parsed


def unmet_count_requirements(counts: dict[str, int], requirements: list[tuple[str, int]]) -> list[str]:
    return [f"{key}={counts.get(key, 0)}<{minimum}" for key, minimum in requirements if counts.get(key, 0) < minimum]


def has_authentic_reasoning(trace) -> bool:
    """Require reasoning attached to a sampled model response, not prompt text."""
    return any(
        node.sampled and node.message.role == "assistant" and bool(getattr(node.message, "reasoning_content", None))
        for node in trace.nodes
    )


def _teacher_conditioned(trace) -> bool:
    return _data_field(trace.task.data, "teacher_conditioned", "false").lower() == "true"


def admission_quality(trace, cohort: Cohort) -> AdmissionQuality | None:
    if not trace.ok or trace.is_truncated:
        return None
    if cohort == "ownership":
        return "clean" if _metric(trace, "strict_success") == 1.0 else None
    if _metric(trace, "answer_accuracy") == 1.0 and _metric(trace, "clean_protocol_aligned") == 1.0:
        return "clean"

    # Bootstrap-only fallback for the asynchronous boundary: preserve a
    # complete, causally correct model trajectory when its sole defect is one
    # redundant local IPython call. This never changes the clean eval gate and
    # is restricted to explicitly teacher-conditioned bidirectional traces.
    family = _data_field(trace.task.data, "family")
    required = {
        "answer_accuracy": 1.0,
        "protocol_aligned": 1.0,
        "natural_followup_causal": 1.0,
        "coordinator_delegated_path_accesses": 0.0,
        "duplicate_cells": 0.0,
        "failed_cells": 0.0,
        "non_ipython_tool_calls": 0.0,
        "observation_calls": 0.0,
        "post_result_coordinator_cells": 0.0,
    }
    if (
        family in {"followup", "handshake"}
        and _teacher_conditioned(trace)
        and all(_metric(trace, name) == expected for name, expected in required.items())
        and (_metric(trace, "bidirectional_control") or 0.0) >= 0.9
        and (_metric(trace, "post_parent_send_tool_calls") or 0.0) <= 1.0
    ):
        return "conditioned_causal_near_clean"
    return None


def admitted(trace, cohort: Cohort) -> bool:
    return admission_quality(trace, cohort) is not None


def coordinator_branches(trace) -> list:
    """The root Prime Agent branch, excluding spawned child branches."""
    if not trace.nodes:
        return []
    return [branch for branch in trace.branches if branch.nodes and branch.nodes[0] is trace.nodes[0]]


def training_rows(trace, cohort: Cohort) -> list[dict]:
    if cohort == "ownership":
        return [sft_row(trace, branch) for branch in coordinator_branches(trace)]
    return sft_rows(trace)


def _data_field(data, name: str, default: str = "unknown") -> str:
    value = getattr(data, name, None)
    if value is None:
        value = (data.model_extra or {}).get(name)
    return str(value) if value is not None else default


def _source_record(run_dir: Path, cohort: Cohort) -> dict:
    traces_path = run_dir / "traces.jsonl"
    config_path = run_dir / "config.toml"
    if not traces_path.is_file() or not config_path.is_file():
        raise SystemExit(f"incomplete eval run: {run_dir}")
    return {
        "cohort": cohort,
        "run_dir": str(run_dir),
        "traces_sha256": _sha256(traces_path),
        "config_sha256": _sha256(config_path),
    }


def build(sources: list[tuple[Path, Cohort]]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    source_records = []
    for run_dir, cohort in sources:
        source_records.append(_source_record(run_dir, cohort))
        for episode in read_episodes(run_dir, WireTrace):
            for trace in episode.traces:
                counts[f"{cohort}.seen"] += 1
                quality = admission_quality(trace, cohort)
                if trace.id in seen or quality is None:
                    continue
                seen.add(trace.id)
                family = _data_field(trace.task.data, "family")
                if family == "unknown" and cohort == "ownership":
                    family = _data_field(trace.task.data, "resource_family")
                task_name = _data_field(trace.task.data, "name")
                ownership = _data_field(trace.task.data, "ownership") if cohort == "ownership" else "unknown"
                instruction_level = _data_field(trace.task.data, "instruction_level", "standard")
                teacher_conditioned = _data_field(trace.task.data, "teacher_conditioned", "false").lower() == "true"
                conditioning = "teacher_conditioned" if teacher_conditioned else "none"
                trace_rows = training_rows(trace, cohort)
                if not trace_rows:
                    raise ValueError(f"admitted trace {trace.id} has no eligible branch")
                for row in trace_rows:
                    row.update(
                        source_cohort=cohort,
                        source_trace_id=trace.id,
                        source_task=task_name,
                        source_family=family,
                        source_ownership=ownership,
                        source_instruction_level=instruction_level,
                        source_conditioning=conditioning,
                        source_admission_quality=quality,
                    )
                rows.extend(trace_rows)
                counts[f"{cohort}.admitted_traces"] += 1
                counts[f"{cohort}.rows"] += len(trace_rows)
                counts[f"family.{family}"] += 1
                counts[f"instruction.{instruction_level}.admitted_traces"] += 1
                counts[f"{cohort}.instruction.{instruction_level}.admitted_traces"] += 1
                counts[f"conditioning.{conditioning}.admitted_traces"] += 1
                counts[f"{cohort}.conditioning.{conditioning}.admitted_traces"] += 1
                counts[f"quality.{quality}.admitted_traces"] += 1
                counts[f"{cohort}.quality.{quality}.admitted_traces"] += 1
                reasoning = "present" if has_authentic_reasoning(trace) else "absent"
                counts[f"reasoning.{reasoning}_traces"] += 1
                counts[f"{cohort}.reasoning.{reasoning}_traces"] += 1
                if cohort == "ownership":
                    counts[f"ownership.{ownership}.admitted_traces"] += 1
                    counts[f"ownership.{ownership}.family.{family}"] += 1
    manifest = {
        "schema_version": 1,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "selection": {
            "ownership": "ok and not truncated and strict_success == 1",
            "communication": ("ok and not truncated and answer_accuracy == 1 and clean_protocol_aligned == 1"),
            "conditioned_bidirectional_bootstrap": (
                "teacher-conditioned followup/handshake only; answer and causal protocol complete; "
                "no ownership, failure, duplicate, observation, or post-result violation; "
                "bidirectional_control >= 0.9 and at most one redundant local IPython call"
            ),
            "promotion": "the conditioned bootstrap exception never changes clean evaluation or teacher promotion gates",
            "reasoning": "preserve sampled reasoning_content when present; never synthesize it",
            "prompts": "preserve the exact admitted trace; never rewrite guided context as standard context",
        },
        "sources": source_records,
        "counts": dict(sorted(counts.items())),
        "rows": len(rows),
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ownership-run", action="append", type=Path, default=[])
    parser.add_argument("--communication-run", action="append", type=Path, default=[])
    parser.add_argument(
        "--require-count",
        action="append",
        type=parse_count_requirement,
        default=[],
        metavar="KEY=MIN",
        help="fail unless the manifest count KEY is at least MIN; repeatable",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="print the prospective manifest without writing a training dataset",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    sources = [
        *((path, "ownership") for path in args.ownership_run),
        *((path, "communication") for path in args.communication_run),
    ]
    if not sources:
        parser.error("at least one source run is required")
    if not args.audit_only and args.output_dir is None:
        parser.error("--output-dir is required unless --audit-only is set")

    rows, manifest = build(sources)
    missing = unmet_count_requirements(manifest["counts"], args.require_count)
    if args.audit_only:
        print(json.dumps({**manifest, "missing_requirements": missing}, indent=2, sort_keys=True))
        raise SystemExit(bool(missing))
    if not rows:
        raise SystemExit("no verified teacher rows were admitted")
    if missing:
        raise SystemExit(f"teacher coverage requirements failed: {', '.join(missing)}")
    assert args.output_dir is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(args.output_dir / "train.parquet"))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["counts"], sort_keys=True))
    print(f"wrote {len(rows)} verified row(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
