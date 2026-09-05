#!/usr/bin/env python3
"""Write a hash-locked forensic receipt for the rejected retained P1 LR=0 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXECUTION_REVISION = "906661c236f4fbb340f3685627199d21b6edd900"
VERIFIERS_REVISION = "87e805c805e92d32638c20ad9ddf878cec880853"
CONFIG_SHA256 = "fa6c43aae8c4865ea2604c74c4dbd7760420e209b7f1640ea0de9aa13b3eac8c"
SAMPLING_SHA256 = "dd6490699885e611b2e9b1ad98e5d7d6d3432866ecf00d7887a3a635998ffd2f"
ROUTE_SHA256 = "2fe2046fdc0cda1f087faf95dc9e48479aa12fd79431641c56e6d7b5eed554c9"
PREFLIGHT_SHA256 = "863732af4abb63753e91ddd57d189d25c1744ea2d647f05f5aed9da4ca3a91a5"
BAD_TRACE_ID = "57f6214886ce4f70a69f3eb2754770ce"
RETAINED_HASHES = {
    "metrics.jsonl": "6c1fbb5804b12e2ecf87201de9929e9ad924174632224385f16d9ecb0d2f9fff",
    "rollouts/step_1/train/all/traces.jsonl": "40ad58939ef11e469ae0a064da9991c21c639dddf891d76fe8051cedb2eb4bc6",
    "rollouts/step_1/train/effective/traces.jsonl": "4a07524637aff3d23fabeaad60f47542873414d99eb09109afe6ca0a0f6a2c14",
    "token_exports/step_1/rank_0.jsonl": "98cb6e750088cba81f900d4e70ebc7dec545e42bb6b4c4942a177d88170aba1d",
    "configs/orchestrator.json": "d0e2521331fc2ed304770ee8bef3d70a6fd3d1dc7923d2415f9255a5d7307579",
    "configs/inference.json": "bce3d2b85e4b703249cc9953de3fa0e16d1b6c056f138e510cae2bb65e50763a",
    "configs/trainer.json": "a18448f2147859a7098311a2dc1730cd6e436c5f270ed6493edd1768943c1393",
    "configs/envs/train/source-worker-ast-s6.json": "0fa2ea6264c36379e872b82436006fc3d3bbe40d4bcf0bb95bf0dd8fa1fb8843",
    "configs/envs/train/source-worker-config-s6.json": "0786029a32262199a716972d90ecf23f83176cd3f7898163e25a8380cb48d03c",
    "final_summary.json": "56ad4ea4fe141452cf2cd933b7d23b2f4f6eb3097e64a62388979d811529d347",
}


class ForensicFailure(ValueError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metric(records: list[dict[str, Any]], name: str) -> float:
    values = [record[name] for record in records if name in record]
    if not values or not isinstance(values[-1], (int, float)) or not math.isfinite(values[-1]):
        raise ForensicFailure(f"P1 metric missing or nonfinite: {name}")
    return float(values[-1])


def group_inventory(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_id = record.get("info", {}).get("group_id")
        if not isinstance(group_id, str):
            raise ForensicFailure("P1 trace lacks group id")
        groups[group_id].append(record)
    rows = []
    for group_id, traces in groups.items():
        rewards = [
            float(trace["rewards"]["source_worker_first_call"]["score"])
            for trace in traces
        ]
        rows.append(
            {
                "group_id": group_id,
                "source": traces[0]["info"]["env_name"],
                "family": traces[0]["task"]["data"]["family"],
                "task_key": traces[0]["task"].get("key") or traces[0]["task"].get("hash"),
                "trace_ids": sorted(trace["id"] for trace in traces),
                "rewards": rewards,
                "population_variance": statistics.pvariance(rewards),
            }
        )
    return sorted(rows, key=lambda row: (row["source"], row["task_key"]))


def _node_null_count(records: list[dict[str, Any]]) -> int:
    return sum(
        call.get("node") is None
        for record in records
        for call in record.get("calls", [])
        if isinstance(call, dict)
    )


def recover(
    run_dir: Path,
    route_audit: Path,
    model_preflight: Path,
    output: Path,
    *,
    audit_revision: str,
) -> dict[str, Any]:
    if len(audit_revision) != 40 or any(character not in "0123456789abcdef" for character in audit_revision):
        raise ForensicFailure("P1 forensic receipt requires an exact audit revision")
    if output.exists() or output.resolve().is_relative_to(run_dir.resolve()):
        raise ForensicFailure("P1 forensic receipt must be new and outside the retained run")
    actual_hashes = {relative: digest(run_dir / relative) for relative in RETAINED_HASHES}
    if actual_hashes != RETAINED_HASHES or digest(route_audit) != ROUTE_SHA256 or digest(model_preflight) != PREFLIGHT_SHA256:
        raise ForensicFailure("P1 retained artifact identity changed")
    raw = read_jsonl(run_dir / "rollouts/step_1/train/all/traces.jsonl")
    effective = read_jsonl(run_dir / "rollouts/step_1/train/effective/traces.jsonl")
    audits = read_jsonl(route_audit)
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    raw_ids = {trace.get("id") for trace in raw}
    effective_ids = {trace.get("id") for trace in effective}
    groups = group_inventory(raw)
    effective_group_ids = {
        trace.get("info", {}).get("group_id") for trace in effective
    }
    rejected_groups = [row for row in groups if row["group_id"] not in effective_group_ids]
    if (
        len(raw) != 24
        or len(effective) != 16
        or not effective_ids < raw_ids
        or len(groups) != 3
        or len(rejected_groups) != 1
        or len(rejected_groups[0]["trace_ids"]) != 8
        or rejected_groups[0]["population_variance"] != 0
        or set(rejected_groups[0]["rewards"]) != {-0.79}
    ):
        raise ForensicFailure("P1 raw/effective zero-variance inventory changed")
    sequences = [record.get("sequence") for record in audits]
    route_counts = Counter(record.get("role") for record in audits)
    bad = next((trace for trace in raw if trace.get("id") == BAD_TRACE_ID), None)
    bad_event = next((event for event in audits if event.get("sequence") == 105), None)
    if (
        len(audits) != 198
        or sequences != list(range(198))
        or route_counts != Counter({"child": 105, "coordinator": 93})
        or bad is None
        or bad_event is None
        or bad_event.get("role") != "coordinator"
        or bad_event.get("upstream_model")
        != "/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2"
    ):
        raise ForensicFailure("P1 route/topology evidence changed")
    model_files = [
        path
        for directory in (run_dir / "checkpoints", run_dir / "weights")
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file()
    ]
    if model_files:
        raise ForensicFailure("P1 unexpectedly contains model artifacts")
    return {
        "schema_version": "q35-2b-source-route-parity-p1-forensic-recovery/v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "verdict": "mechanism_rejected",
        "execution_revision": EXECUTION_REVISION,
        "verifiers_revision": VERIFIERS_REVISION,
        "audit_revision": audit_revision,
        "immutable_inputs": {
            "run_artifacts": actual_hashes,
            "routing_audit_sha256": ROUTE_SHA256,
            "model_preflight_sha256": PREFLIGHT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "sampling_contract_sha256": SAMPLING_SHA256,
        },
        "selection": {
            "raw": 24,
            "effective": 16,
            "groups": groups,
            "rejected_zero_variance_groups": rejected_groups,
            "pre_filter_dropped_rate": metric(metrics, "pre_filters/all/dropped_rate"),
            "pre_filter_zero_advantage_rate": metric(metrics, "pre_filters/all/zero_advantage/rate"),
        },
        "mechanism_failures": {
            "raw_effective_identity_gate": "failed_24_vs_16",
            "terminal_node_null_residues": {
                "raw": _node_null_count(raw),
                "effective": _node_null_count(effective),
                "frozen_limit": "at_most_1_and_at_most_1_percent",
            },
            "terminal_worker_descendant": {
                "trace_id": BAD_TRACE_ID,
                "effective": False,
                "third_call_bearing_root": True,
                "audit_sequence": 105,
                "routed_role": "coordinator",
                "routed_model": "e33",
            },
        },
        "recovered_measurements": {
            "gradient_norm": metric(metrics, "optim/grad_norm"),
            "active_rl_tokens": int(metric(metrics, "loss_tokens/rl")),
            "unmasked_mismatch_mean": metric(metrics, "unmasked_mismatch_kl/mean"),
            "mismatch_max": metric(metrics, "mismatch_kl/all/max"),
            "mask_fraction": metric(metrics, "is_masked/mean"),
            "routes": {"total": len(audits), **dict(route_counts)},
        },
        "interpretation": {
            "launcher_postflight_passed": False,
            "p1_mechanism_admitted": False,
            "calibration_measurements_recovered": True,
            "checkpoint_written": False,
            "optimizer_steps_executed": 1,
            "learning_rate": 0.0,
            "model_weight_change_expected": False,
            "optimizer_update_authorized": False,
            "heldout_evaluation_run": False,
            "next_step_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--route-audit", type=Path, required=True)
    parser.add_argument("--model-preflight", type=Path, required=True)
    parser.add_argument("--audit-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = recover(
            args.run_dir,
            args.route_audit,
            args.model_preflight,
            args.output,
            audit_revision=args.audit_revision,
        )
    except (ForensicFailure, json.JSONDecodeError, OSError, ValueError) as error:
        raise SystemExit(f"P1 forensic recovery failed: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
