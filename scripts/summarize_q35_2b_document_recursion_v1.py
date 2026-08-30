"""Build a zero-update document recursion capability receipt."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from summarize_prime_agent_mastery_v2 import load_traces, summarize

TOPOLOGIES = ("direct", "flat", "hierarchical")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"missing routing audit: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SystemExit(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _routing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    role_counts = Counter(str(row.get("role")) for row in rows)
    sessions: dict[str, set[str]] = {"coordinator": set(), "child": set()}
    for row in rows:
        role = row.get("role")
        session = row.get("session_sha256")
        if role in sessions and isinstance(session, str):
            sessions[role].add(session)
    return {
        "request_count": len(rows),
        "role_requests": dict(sorted(role_counts.items())),
        "unique_sessions": {role: len(values) for role, values in sessions.items()},
    }


def _require_routing(topology: str, routing: dict[str, Any]) -> None:
    sessions = routing["unique_sessions"]
    if sessions["coordinator"] < 4:
        raise SystemExit(f"{topology}: fewer than four root coordinator sessions were routed")
    if topology == "direct":
        if sessions["child"] != 0:
            raise SystemExit("direct: child policy received a request")
        return
    if sessions["child"] < 1:
        raise SystemExit(f"{topology}: child policy received no request")
    if topology == "hierarchical" and sessions["coordinator"] < 5:
        raise SystemExit("hierarchical: no distinct non-root coordinator session was routed")


def build_receipt(
    output_root: Path,
    label: str,
    coordinator_sha256: str,
    child_sha256: str,
) -> dict[str, Any]:
    topologies = {}
    for topology in TOPOLOGIES:
        run_root = output_root / f"{label}-{topology}"
        traces = load_traces([run_root / "document" / "document"])
        if len(traces) != 4:
            raise SystemExit(f"{topology}: expected four traces, found {len(traces)}")
        trace_summary = summarize(traces)
        routing = _routing_summary(_load_jsonl(run_root / "ROUTING_AUDIT.jsonl"))
        _require_routing(topology, routing)
        topologies[topology] = {
            "trace_summary": trace_summary,
            "routing": routing,
        }
    return {
        "experiment": "q35-2b-document-recursion-zero-update-v1",
        "zero_optimizer_updates": True,
        "weights_unchanged": True,
        "coordinator_model_sha256": coordinator_sha256,
        "child_model_sha256": child_sha256,
        "topologies": topologies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--coordinator-sha256", required=True)
    parser.add_argument("--child-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        args.output_root,
        args.label,
        args.coordinator_sha256,
        args.child_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
