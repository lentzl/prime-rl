#!/usr/bin/env python3
"""Write a hash-locked rejection receipt for the retained P2R LR=0 run."""

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

EXECUTION_REVISION = "c5af5ad0bc7fe89e7a256804d1e7cc2591e1a612"
VERIFIERS_REVISION = "c1a2f5bf3db3f34206e45b04442e64ca6a7770de"
CONFIG_SHA256 = "dba4062e27c6b4395c37c6a1477bd4fd6edf549b6e622bc174fe95add1fffabf"
SAMPLING_SHA256 = "26f7c37076e178528358357fc3a6840455e33b4e639fb1e4fac48c44601aca74"
TASK_BANK_SHA256 = "cb67bfe68866e0ec355159a39e8dea5fed92ba5f7f0c4e7b5185e6371088c5db"
TASK_KEY_SET_SHA256 = "f0e577eb1872d0ad0b6bbd8208d09448de3b8ab65ae0e4da92310ae70f59284e"
SNAPSHOT_MANIFEST_SHA256 = "538ecc2007ec25cc050993aa6ccd47b959f79248634d7e9ac79695a3d45e69c9"
RETAINED_HASHES = {
    "configs/envs/train/source-worker-ast-s6.json": "18b6b0190a8b0c0ba2c8c5422cb1849cd5c47a8920b940d24172ab9dc933f955",
    "configs/envs/train/source-worker-config-s6.json": "1001ba726a2d1c33114a2c89a486b771b857e88046c6b9abb6c3c8921e61d263",
    "configs/inference.json": "36e6e1df0d9bff2cea4032808d9bfbf17a1bef4b4f618f5e628205f68e0a4504",
    "configs/orchestrator.json": "4e06df04274bead579ce44f2ebe90f20be3282ea44a92a3625e11e4d5ede7e21",
    "configs/trainer.json": "ea9cd61915848710446ae33c2d53f677d002778ffbaeb5460bec85a0a6ca6cd7",
    "metrics.jsonl": "76116297c4e68f12c88462d595b23ce5f9c7aaa8ea5bc4a9174cec334a6adb1b",
    "rollouts/step_1/train/all/traces.jsonl": "b6408adedb7865ace065909c9afdc60fbfa8f0ea6a57656fb2be601d449e0974",
    "rollouts/step_1/train/effective/traces.jsonl": "f7a59c0e12da0002f4f5c766f3b88146c769cb828b12a945f40fbf8055610bc5",
    "token_exports/step_1/STABLE": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "token_exports/step_1/rank_0.jsonl": "9b6f9861a94e71817847c16f02f11ca7894d2499238c033b48c2a53fdd565c53",
}
RESULT_HASHES = {
    "core-verifiers-import-provenance.json": "5546ce5f363685218caec5df2a51573cc5873b5d23a43043d6e54cdcaa109aa0",
    "postflight-model-hashes.json": "2ccf9ea70ae5855457bd549d5389ea8d934f79a62618010ca7b3228aeb1931a1",
    "preflight-model-hashes.json": "1c400f31954c4bfb6a455fb2ed2add08304a7711368f532bf018f5cb8ac72c0c",
    "routing-audit.jsonl": "bd093a5bd42606eaa0480ffd7db412a6c95181d7b726fec91c25c3b7d47a9124",
}
EXPECTED_MODELS = {
    "H176": "77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e",
    "S2": "937a96154cd47d8dda1fb01125d9f552037a91bc0542748f417652ceddd47f47",
    "S5": "09a2e3e88030d17896e554211d8fc7eff709d6b4a619e99e3342d05aacde0782",
    "e33": "e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47",
}
EXPECTED_UPSTREAM = {
    "coordinator": "/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2",
    "child": "/home/ubuntu/rlm/outputs/q35-2b-source-worker-remedial-s5-v1/h-source-s5-remedial-step8-v1/weights/step_8",
}
FAMILIES = ("specialist_source_ast", "specialist_source_config")
SOURCES = {
    "specialist_source_ast": "source-worker-ast-s6",
    "specialist_source_config": "source-worker-config-s6",
}
SPECIALIST_MARKERS = (
    "[task from parent]",
    "[selected terminal capability]",
    "expert_id=source_inspector",
    "session_role=terminal_worker",
)


class ForensicFailure(ValueError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ForensicFailure(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ForensicFailure(f"expected JSON objects: {path}")
    return rows


def metric(records: list[dict[str, Any]], name: str) -> float:
    values = [record[name] for record in records if name in record]
    if not values or not isinstance(values[-1], (int, float)) or not math.isfinite(values[-1]):
        raise ForensicFailure(f"P2R metric missing or nonfinite: {name}")
    return float(values[-1])


def _wire_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in value
        )
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return ""


def _root(nodes: list[dict[str, Any]], index: int) -> int:
    seen: set[int] = set()
    while index not in seen:
        seen.add(index)
        parent = nodes[index].get("parent")
        if parent is None:
            return index
        if not isinstance(parent, int) or not 0 <= parent < len(nodes):
            raise ForensicFailure("P2R trace has invalid node ancestry")
        index = parent
    raise ForensicFailure("P2R trace has cyclic node ancestry")


def _call_role(nodes: list[dict[str, Any]], index: int) -> str:
    seen: set[int] = set()
    while index not in seen:
        seen.add(index)
        node = nodes[index]
        message = node.get("message")
        if isinstance(message, dict):
            text = _wire_text(message.get("content"))
            if text.lstrip().startswith(SPECIALIST_MARKERS[0]):
                if not all(marker in text for marker in SPECIALIST_MARKERS):
                    raise ForensicFailure("P2R child marker is incomplete")
                return "child"
        parent = node.get("parent")
        if parent is None:
            return "coordinator"
        if not isinstance(parent, int) or not 0 <= parent < len(nodes):
            raise ForensicFailure("P2R trace has invalid node ancestry")
        index = parent
    raise ForensicFailure("P2R trace has cyclic node ancestry")


def _trace_route_inventory(
    traces: list[dict[str, Any]], audits: list[dict[str, Any]]
) -> dict[str, Any]:
    expected: Counter[tuple[str, str]] = Counter()
    residues = []
    all_branch_hashes: set[str] = set()
    trace_roles: Counter[str] = Counter()
    for trace in traces:
        nodes = trace.get("nodes")
        calls = trace.get("calls")
        if not isinstance(nodes, list) or not isinstance(calls, list):
            raise ForensicFailure("P2R trace lacks nodes/calls")
        attached_sessions: dict[str, str] = {}
        roots: dict[int, tuple[str, str]] = {}
        for call in calls:
            node_index = call.get("node")
            client_session = call.get("client_session_id")
            if not isinstance(client_session, str) or not client_session:
                raise ForensicFailure("P2R call lacks client session")
            branch_hash = hashlib.sha256(client_session.encode()).hexdigest()
            all_branch_hashes.add(branch_hash)
            if isinstance(node_index, int):
                role = _call_role(nodes, node_index)
                root_index = _root(nodes, node_index)
                prior = roots.setdefault(root_index, (branch_hash, role))
                if prior != (branch_hash, role):
                    raise ForensicFailure("P2R graph root maps to multiple branches")
                attached_sessions[client_session] = role
            elif node_index is not None:
                raise ForensicFailure("P2R call node is malformed")
        if len(roots) != 2 or Counter(role for _, role in roots.values()) != Counter(
            {"coordinator": 1, "child": 1}
        ):
            raise ForensicFailure("P2R trace is not one coordinator plus one terminal root")
        if any("Recursive agent depth: 2" in _wire_text(node.get("message", {}).get("content")) for node in nodes if isinstance(node, dict) and isinstance(node.get("message"), dict)):
            raise ForensicFailure("P2R terminal worker spawned a descendant")
        trace_null = 0
        for index, call in enumerate(calls):
            client_session = call["client_session_id"]
            role = attached_sessions.get(client_session)
            if role is None:
                raise ForensicFailure("P2R terminal call has no attached branch identity")
            branch_hash = hashlib.sha256(client_session.encode()).hexdigest()
            expected[(branch_hash, role)] += 1
            trace_roles[role] += 1
            if call.get("node") is None:
                trace_null += 1
                usage = call.get("usage")
                tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
                if (
                    index != len(calls) - 1
                    or trace.get("stop_condition") != "max_turns"
                    or call.get("error") is not None
                    or not isinstance(tokens, int)
                ):
                    raise ForensicFailure("P2R node-null call is not an exact terminal residue")
                residues.append(
                    {
                        "trace_id": trace.get("id"),
                        "role": role,
                        "branch_session_sha256": branch_hash,
                        "wasted_completion_tokens": tokens,
                    }
                )
        if trace_null > 1:
            raise ForensicFailure("P2R trace contains multiple terminal residues")
    observed = Counter((row.get("branch_session_sha256"), row.get("role")) for row in audits)
    if expected != observed:
        raise ForensicFailure("P2R call/route branch-role multiplicity differs")
    return {
        "trace_calls": sum(expected.values()),
        "attached_successful_calls": {
            role: trace_roles[role] - sum(row["role"] == role for row in residues)
            for role in ("coordinator", "child")
        },
        "route_counts_including_terminal_residues": dict(trace_roles),
        "branch_session_hashes": len(all_branch_hashes),
        "terminal_residues": residues,
        "terminal_residue_fraction": len(residues) / sum(expected.values()),
        "wasted_completion_tokens": sum(row["wasted_completion_tokens"] for row in residues),
        "terminal_worker_descendants": 0,
        "exact_branch_role_multiplicity": True,
    }


def _export_inventory(exports: list[dict[str, Any]]) -> dict[str, Any]:
    active: Counter[str] = Counter()
    masked: Counter[str] = Counter()
    mismatch: dict[str, list[float]] = defaultdict(list)
    entropy: dict[str, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in exports:
        source = row.get("env_name")
        if source not in SOURCES.values():
            raise ForensicFailure("P2R export has unexpected source")
        counts[source] += 1
        fields = [row.get(name) for name in ("loss_mask", "rl_weights", "is_masked", "mismatch_kl", "entropy")]
        if not all(isinstance(value, list) for value in fields) or len({len(value) for value in fields}) != 1:
            raise ForensicFailure("P2R export has malformed token evidence")
        loss_mask, weights, is_masked, mismatch_kl, entropies = fields
        for keep, weight, masked_value, mismatch_value, entropy_value in zip(
            loss_mask, weights, is_masked, mismatch_kl, entropies, strict=True
        ):
            if not keep or (weight is not None and float(weight) == 0.0):
                continue
            active[source] += 1
            if masked_value:
                masked[source] += 1
                continue
            if not isinstance(mismatch_value, (int, float)) or not isinstance(entropy_value, (int, float)):
                raise ForensicFailure("P2R active token lacks likelihood evidence")
            mismatch[source].append(float(mismatch_value))
            entropy[source].append(float(entropy_value))
    if counts != Counter({source: 8 for source in SOURCES.values()}):
        raise ForensicFailure("P2R exports are not exactly 8 per source")
    by_source = {}
    for source in SOURCES.values():
        by_source[source] = {
            "exports": counts[source],
            "active_rl_tokens": active[source],
            "masked_active_tokens": masked[source],
            "mask_fraction": masked[source] / active[source],
            "unmasked_mismatch_mean": statistics.fmean(mismatch[source]),
            "active_mismatch_max_abs": max(map(abs, mismatch[source])),
            "unmasked_entropy_mean": statistics.fmean(entropy[source]),
            "active_entropy_max": max(entropy[source]),
            "mean_mismatch_to_entropy_ratio": abs(statistics.fmean(mismatch[source]))
            / statistics.fmean(entropy[source]),
            "max_mismatch_to_entropy_ratio": max(map(abs, mismatch[source]))
            / max(entropy[source]),
        }
    return {
        "exports": len(exports),
        "active_rl_tokens": sum(active.values()),
        "masked_active_tokens": sum(masked.values()),
        "mask_fraction": sum(masked.values()) / sum(active.values()),
        "by_source": by_source,
    }


def recover(
    run_dir: Path,
    result_dir: Path,
    snapshot_manifest: Path,
    output: Path,
    *,
    audit_revision: str,
) -> dict[str, Any]:
    if len(audit_revision) != 40 or any(character not in "0123456789abcdef" for character in audit_revision):
        raise ForensicFailure("P2R recovery requires an exact audit revision")
    if audit_revision == EXECUTION_REVISION:
        raise ForensicFailure("P2R recovery revision must differ from execution")
    if output.exists() or output.resolve().is_relative_to(run_dir.resolve()) or output.resolve().is_relative_to(result_dir.resolve()):
        raise ForensicFailure("P2R recovery output must be new and outside retained artifacts")
    if digest(snapshot_manifest) != SNAPSHOT_MANIFEST_SHA256:
        raise ForensicFailure("P2R snapshot manifest identity changed")
    run_hashes = {relative: digest(run_dir / relative) for relative in RETAINED_HASHES}
    result_hashes = {relative: digest(result_dir / relative) for relative in RESULT_HASHES}
    if run_hashes != RETAINED_HASHES or result_hashes != RESULT_HASHES:
        raise ForensicFailure("P2R retained artifact identity changed")

    raw = read_jsonl(run_dir / "rollouts/step_1/train/all/traces.jsonl")
    effective = read_jsonl(run_dir / "rollouts/step_1/train/effective/traces.jsonl")
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    exports = read_jsonl(run_dir / "token_exports/step_1/rank_0.jsonl")
    audits = read_jsonl(result_dir / "routing-audit.jsonl")
    if (
        len(raw) != 16
        or len(effective) != 16
        or {row.get("id") for row in raw} != {row.get("id") for row in effective}
    ):
        raise ForensicFailure("P2R raw/effective identity changed")
    sequences = [row.get("sequence") for row in audits]
    if (
        len(audits) != 133
        or sequences != list(range(133))
        or any(row.get("status") != 200 for row in audits)
        or len({row.get("request_sha256") for row in audits}) != 133
        or len({row.get("session_sha256") for row in audits}) != 16
        or any(not isinstance(row.get("branch_session_sha256"), str) for row in audits)
    ):
        raise ForensicFailure("P2R routing audit identity/cardinality changed")
    route_inventory = _trace_route_inventory(raw, audits)
    if (
        route_inventory["trace_calls"] != 133
        or route_inventory["branch_session_hashes"] != 32
        or route_inventory["terminal_residue_fraction"] > 0.04
        or route_inventory["wasted_completion_tokens"] != 825
    ):
        raise ForensicFailure("P2R route/topology evidence changed")
    route_roles = Counter(row.get("role") for row in audits)
    forced = [row for row in audits if row.get("mode") == "forced_specialist_assignment_generate_action"]
    if (
        route_roles != Counter({"child": 68, "coordinator": 65})
        or len(forced) != 16
        or any(row.get("role") != "coordinator" for row in forced)
        or any(row.get("upstream_model") != EXPECTED_UPSTREAM[row["role"]] for row in audits)
    ):
        raise ForensicFailure("P2R exact upstream role routing changed")

    family_rows = {
        family: [row for row in effective if row.get("task", {}).get("data", {}).get("family") == family]
        for family in FAMILIES
    }
    if any(len(rows) != 8 for rows in family_rows.values()):
        raise ForensicFailure("P2R effective family balance changed")
    stops = {family: Counter(row.get("stop_condition") for row in rows) for family, rows in family_rows.items()}
    rewards = {
        family: [float(row["rewards"]["source_worker_first_call"]["score"]) for row in rows]
        for family, rows in family_rows.items()
    }
    components = (
        "exception_free_first_call",
        "correct_file_api",
        "exact_oracle_value",
        "atomic_compact_parent_send",
        "protocol_aligned",
        "clean_protocol_aligned",
        "messages_to_parent",
        "failed_cells",
        "retries",
        "extra_sends",
        "answer_accuracy",
    )
    competence = {
        family: {
            name: sum(float(row.get("metrics", {}).get(name, 0.0)) for row in rows)
            for name in components
        }
        for family, rows in family_rows.items()
    }
    export_inventory = _export_inventory(exports)
    if export_inventory["active_rl_tokens"] != int(metric(metrics, "loss_tokens/rl")):
        raise ForensicFailure("P2R export and trainer token mass differ")
    if any(metric(metrics, f"loss_tokens/{name}") != 0 for name in ("ce", "ref_kl", "sdpo")):
        raise ForensicFailure("P2R non-RL loss received token mass")

    preflight = read_json(result_dir / "preflight-model-hashes.json")
    postflight = read_json(result_dir / "postflight-model-hashes.json")
    pre_models = preflight.get("models")
    post_models = postflight.get("models")
    if (
        pre_models != post_models
        or not isinstance(pre_models, dict)
        or {name: row.get("model_sha256") for name, row in pre_models.items()} != EXPECTED_MODELS
        or postflight.get("learning_rate") != 0.0
        or postflight.get("checkpoint_written") is not False
        or postflight.get("execution_revision") != EXECUTION_REVISION
        or postflight.get("verifiers_revision") != VERIFIERS_REVISION
        or postflight.get("config_sha256") != CONFIG_SHA256
    ):
        raise ForensicFailure("P2R protected model evidence changed")
    model_files = [
        path
        for directory in (run_dir / "weights", run_dir / "checkpoints")
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file()
    ]
    if model_files:
        raise ForensicFailure("P2R unexpectedly contains model artifacts")

    no_drop_keys = (
        "train/agg/all/agent/filters/zero_advantage/mean",
        "train/agg/effective/agent/filters/zero_advantage/mean",
        "train/source-worker-ast-s6/all/agent/filters/zero_advantage/mean",
        "train/source-worker-ast-s6/effective/agent/filters/zero_advantage/mean",
        "train/source-worker-config-s6/all/agent/filters/zero_advantage/mean",
        "train/source-worker-config-s6/effective/agent/filters/zero_advantage/mean",
    )
    if (
        any("pre_filters/all/zero_advantage/rate" in row for row in metrics)
        or metric(metrics, "pre_filters/all/dropped_rate") != 0.0
        or any(metric(metrics, name) != 0.0 for name in no_drop_keys)
    ):
        raise ForensicFailure("P2R no-drop metric schema evidence changed")

    ast_export = export_inventory["by_source"][SOURCES["specialist_source_ast"]]
    config_export = export_inventory["by_source"][SOURCES["specialist_source_config"]]
    return {
        "schema_version": "q35-2b-source-route-parity-p2r-forensic-recovery/v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "verdict": "mechanism_rejected",
        "execution_revision": EXECUTION_REVISION,
        "verifiers_revision": VERIFIERS_REVISION,
        "audit_revision": audit_revision,
        "immutable_inputs": {
            "snapshot_manifest_sha256": SNAPSHOT_MANIFEST_SHA256,
            "run_artifacts": run_hashes,
            "result_artifacts": result_hashes,
            "config_sha256": CONFIG_SHA256,
            "sampling_contract_sha256": SAMPLING_SHA256,
            "task_bank_sha256": TASK_BANK_SHA256,
            "task_key_set_sha256": TASK_KEY_SET_SHA256,
        },
        "structural_evidence": {
            "raw": len(raw),
            "effective": len(effective),
            "raw_effective_identity_equal": True,
            "families": {family: len(rows) for family, rows in family_rows.items()},
            "routing": {
                **route_inventory,
                "route_events": len(audits),
                "route_events_by_role": dict(route_roles),
                "forced_coordinator_events": len(forced),
                "rollout_sessions": 16,
                "status_200": len(audits),
                "audit_only_events": 0,
            },
            "exports": export_inventory,
            "child_branches": 16,
            "coordinator_exported_trainable_tokens": 0,
            "non_rl_loss_tokens": 0,
        },
        "reward_and_behavior": {
            "stops": {family: dict(counts) for family, counts in stops.items()},
            "rewards": {
                family: {
                    "values": values,
                    "population_variance": statistics.pvariance(values),
                    "range": max(values) - min(values),
                }
                for family, values in rewards.items()
            },
            "competence_components": competence,
            "interpretation": "One config child sent a later exact body; no trace made a useful exception-free designated first call, and zero-retry AST silence tied for best AST reward.",
        },
        "optimizer_and_likelihood": {
            "learning_rate": metric(metrics, "optim/lr"),
            "optimizer_steps_executed": 1,
            "update_succeeded": metric(metrics, "optim/update_succeeded"),
            "gradient_norm": metric(metrics, "optim/grad_norm"),
            "aggregate_entropy_mean": metric(metrics, "entropy/all/mean"),
            "aggregate_mismatch_mean": metric(metrics, "unmasked_mismatch_kl/mean"),
            "aggregate_metric_mismatch_max": metric(metrics, "mismatch_kl/all/max"),
            "export_derived_ast": ast_export,
            "export_derived_config": config_export,
            "mask_fraction": metric(metrics, "is_masked/mean"),
            "checkpoint_written": False,
            "protected_models_unchanged": True,
        },
        "mechanism_failures": {
            "config_natural_termination": {
                "observed": dict(stops["specialist_source_config"]),
                "required": "at least one agent_completed and at most seven max_turns",
            },
            "ast_absolute_mismatch_max": {
                "observed": ast_export["active_mismatch_max_abs"],
                "maximum": 0.35,
            },
            "ast_mismatch_to_entropy_max_ratio": {
                "observed": ast_export["max_mismatch_to_entropy_ratio"],
                "maximum": 0.05,
            },
            "no_useful_first_call_support": {
                "exception_free": 0,
                "correct_api": 0,
                "atomic": 0,
                "silent_ast_reward": 0.0,
            },
            "validator_schema_gap": {
                "missing_metric": "pre_filters/all/zero_advantage/rate",
                "corroborating_raw_equals_effective": True,
                "corroborating_dropped_rate": 0.0,
                "corroborating_zero_filter_monitors": 0.0,
                "scientific_effect": "none; independent frozen health failures reject P2R",
            },
        },
        "interpretation": {
            "launcher_postflight_passed": False,
            "p2r_mechanism_admitted": False,
            "calibration_measurements_recovered": True,
            "optimizer_update_authorized": False,
            "heldout_evaluation_run": False,
            "next_step_authorized": False,
            "downstream_gate_changed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--audit-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = recover(
            args.run_dir,
            args.result_dir,
            args.snapshot_manifest,
            args.output,
            audit_revision=args.audit_revision,
        )
    except (ForensicFailure, json.JSONDecodeError, OSError, ValueError) as error:
        raise SystemExit(f"P2R forensic recovery failed: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
