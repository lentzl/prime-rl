#!/usr/bin/env python3
"""Validate and record the P0 route-parity LR=0 calibration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "validate_q35_2b_source_first_call_grpo_s6_v1.py"
SPEC = importlib.util.spec_from_file_location("source_first_call_s6_validator", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the shared S6 structural validator")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

P0_CONFIG = Path(
    "experiments/qwen35-2b-document-recursion-zero-update-v1/"
    "specialist-source-route-parity-p0-lr0.toml"
)
P0_OUTPUT_DIR = Path(
    "/home/ubuntu/rlm/outputs/q35-2b-source-route-parity-p0-v1/lr0-calibration"
)
P0_RUN_DIR = P0_OUTPUT_DIR / "source-route-parity-p0-lr0-calibration"
P0_RESULT_ROOT = Path(
    "/home/ubuntu/rlm/results/q35-2b-source-route-parity-p0-v1"
)
P0_ROUTING_AUDIT = P0_RESULT_ROOT / "routing-audit.jsonl"
EXPECTED_MODEL_HASHES = {
    "e33": "e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47",
    "H176": "77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e",
    "S2": "937a96154cd47d8dda1fb01125d9f552037a91bc0542748f417652ceddd47f47",
    "S5": "09a2e3e88030d17896e554211d8fc7eff709d6b4a619e99e3342d05aacde0782",
}
EXPECTED_MODEL_PATHS = {
    "e33": BASE.E33_PATH,
    "H176": Path(
        "/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/"
        "h176child8-document-child-real12-step8-v2/weights/step_8"
    ),
    "S2": Path(
        "/home/ubuntu/rlm/outputs/q35-2b-specialist-competence-s2-v1/"
        "h-source-s2-step8-v1/weights/step_8"
    ),
    "S5": BASE.S5_PATH,
}
P0_EXECUTION_REVISION = "35f9cd4667132902a98ed41b494ca44253635022"
P0_CONFIG_SHA256 = "5e21a0d1906e35353e05c0d40367d230614f920fa82cfd27e9b2ba60ad076390"
P0_RETAINED_ARTIFACT_HASHES = {
    "metrics.jsonl": "7cad60b489d72318cb443e181485bb96373df08c59a3d3b30d00fef9ecfaf14c",
    "rollouts/step_1/train/all/traces.jsonl": "3caece22684808a7a7dcceeb93b7763fca2528df97a0707851e1ad33dd25a16b",
    "rollouts/step_1/train/effective/traces.jsonl": "3f27fb44d03eb965d2242cff536c1927d669f4883e6bc18c5d9b9ad4ef80290b",
    "token_exports/step_1/rank_0.jsonl": "e32ebc439dd08fafba88bfe936043170e291cf7bc2ecd8d490d24e60a43258f7",
    "configs/orchestrator.json": "c3d8967ae047f1f53dcb258a95f193f0f88ecaf3294d0949b36833d122d33457",
    "configs/inference.json": "a897cd621240183291dac4ce76360645d25129ddb1741ff713ed010971cd4a73",
    "configs/trainer.json": "af66d4b322f88abb8d148e8ef59de042304a494854500e29fc2fa4b4bca8856d",
}
P0_RETAINED_ROUTING_AUDIT_SHA256 = (
    "a1d5bb71177400262abd7324ad167c39856768499b6317009a4b1556097c29a6"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(path: Path) -> dict[str, Any]:
    report = BASE.validate_config(path, "audit")
    payload = tomllib.loads(path.read_text())
    run = payload.get("run", {})
    router = payload.get("inference", {}).get("router", {})
    if path != P0_CONFIG:
        raise BASE.AuditFailure("P0 requires its unique prospective config path")
    if (
        payload.get("output_dir") != str(P0_OUTPUT_DIR)
        or run.get("name") != P0_RUN_DIR.name
        or run.get("dir") != P0_RUN_DIR.name
        or router.get("state_dir") != str(P0_RESULT_ROOT / "role-router")
        or router.get("audit_log") != str(P0_ROUTING_AUDIT)
    ):
        raise BASE.AuditFailure("P0 output, run, or route-audit identity changed")
    if "eval" in payload or any(
        isinstance(section, dict) and "ckpt" in section
        for section in (
            payload,
            payload.get("trainer", {}),
            payload.get("orchestrator", {}),
        )
    ):
        raise BASE.AuditFailure("P0 forbids evaluation and checkpoint configuration")
    return {
        **report,
        "experiment": "P0 route-parity calibration",
        "output_dir": str(P0_OUTPUT_DIR),
        "run_dir": str(P0_RUN_DIR),
        "routing_audit": str(P0_ROUTING_AUDIT),
        "observational_only": True,
        "thresholds_evaluated": False,
        "optimizer_update_authorized": False,
    }


def _reward_observations(trace_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[float]] = {
        family: [] for family in sorted(BASE.EXPECTED_FAMILIES)
    }
    rows = []
    for index, record in enumerate(trace_records):
        family = record.get("task", {}).get("data", {}).get("family")
        reward = record.get("rewards", {}).get(BASE.EXPECTED_REWARD, {}).get("score")
        if family not in by_family or not isinstance(reward, (int, float)):
            raise BASE.AuditFailure(
                f"effective trace {index} lacks finite P0 reward evidence"
            )
        value = float(reward)
        if not math.isfinite(value):
            raise BASE.AuditFailure(
                f"effective trace {index} has non-finite P0 reward evidence"
            )
        by_family[family].append(value)
        rows.append(
            {
                "trace_id": record.get("id"),
                "task_key": record.get("task", {}).get("key")
                or record.get("task", {}).get("hash"),
                "family": family,
                "reward": value,
            }
        )

    def summary(values: list[float]) -> dict[str, Any]:
        return {
            "count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "mean": statistics.fmean(values),
            "population_variance": statistics.pvariance(values),
            "unique_values": sorted(set(values)),
        }

    all_values = [row["reward"] for row in rows]
    return {
        "overall": summary(all_values),
        "by_family": {
            family: summary(values) for family, values in by_family.items()
        },
        "traces": rows,
        "variance_is_observational": True,
    }


def validate_runtime(
    run_dir: Path,
    *,
    execution_revision: str,
    verifiers_revision: str,
    config_sha256: str,
) -> dict[str, Any]:
    if run_dir != P0_RUN_DIR:
        raise BASE.AuditFailure("P0 runtime path is not the unique write-once run")
    if _digest(P0_CONFIG) != config_sha256:
        raise BASE.AuditFailure("P0 config hash changed after launch")
    structural = BASE.validate_runtime(
        run_dir,
        "audit",
        calibration_only=True,
        routing_audit_path=P0_ROUTING_AUDIT,
    )
    trace_path = (
        run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
    )
    trace_records = BASE._read_jsonl(trace_path)
    metrics_path = run_dir / "metrics.jsonl"
    token_exports = run_dir / "token_exports" / "step_1"
    if any(
        path.is_file()
        for directory in (run_dir / "checkpoints", run_dir / "weights")
        if directory.exists()
        for path in directory.rglob("*")
    ):
        raise BASE.AuditFailure("P0 wrote a forbidden checkpoint or weight artifact")
    return {
        "schema_version": "q35-2b-source-route-parity-p0-calibration/v1",
        "recorded_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "verdict": "structural_safety_pass",
        "experiment": "P0 route-parity LR=0 calibration",
        "execution_revision": execution_revision,
        "verifiers_revision": verifiers_revision,
        "config": {
            "path": str(P0_CONFIG),
            "sha256": config_sha256,
            "train_seed": BASE.EXPECTED_TRAIN_SEED,
            "instance_offset": BASE.EXPECTED_TRAIN_OFFSET,
        },
        "models": {
            label: {
                "path": str(EXPECTED_MODEL_PATHS[label]),
                "model_sha256": digest,
                "protected_pre_and_post": True,
            }
            for label, digest in EXPECTED_MODEL_HASHES.items()
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "metrics": {"path": str(metrics_path), "sha256": _digest(metrics_path)},
            "effective_traces": {
                "path": str(trace_path),
                "sha256": _digest(trace_path),
            },
            "routing_audit": {
                "path": str(P0_ROUTING_AUDIT),
                "sha256": _digest(P0_ROUTING_AUDIT),
            },
            "resolved_configs": {
                name: {
                    "path": str(run_dir / "configs" / f"{name}.json"),
                    "sha256": _digest(run_dir / "configs" / f"{name}.json"),
                }
                for name in ("orchestrator", "inference", "trainer")
            },
            "token_exports_stable": (token_exports / "STABLE").is_file(),
        },
        "reward": _reward_observations(trace_records),
        "termination_and_censoring": structural["prospective_lr0_health"],
        "trainer_inference": structural["prospective_lr0_health"]["metrics"],
        "token_mass": {
            "child_branches": structural["child_branches"],
            "child_trainable_tokens": structural["child_trainable_tokens"],
            "exported_rl_tokens": structural["exported_rl_tokens"],
            "raw_coordinator_sampled_tokens": structural[
                "raw_coordinator_sampled_tokens"
            ],
            "coordinator_exported_trainable_tokens": structural[
                "coordinator_exported_trainable_tokens"
            ],
            "gradient_norm": structural["gradient_norm"],
        },
        "routing": {
            "forced_assignments": structural["routing"],
            "effective_calls": structural["effective_call_routes"],
            "trace_sets": structural["trace_sets"],
        },
        "interpretation": {
            "observational_only": True,
            "numeric_thresholds_evaluated": False,
            "numeric_thresholds_frozen": False,
            "checkpoint_written": False,
            "optimizer_steps_executed": 1,
            "learning_rate": 0.0,
            "persisted_optimizer_state": False,
            "model_weight_change_expected": False,
            "model_weight_identity_verified_pre_and_post": True,
            "optimizer_update_authorized": False,
            "heldout_evaluation_run": False,
            "next_step_authorized": False,
        },
    }


def recover_retained_runtime(
    run_dir: Path,
    routing_audit_read_path: Path,
    *,
    execution_revision: str,
    audit_revision: str,
    verifiers_revision: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Create a post-hoc forensic receipt for the immutable retained P0 run."""

    if (
        execution_revision != P0_EXECUTION_REVISION
        or config_sha256 != P0_CONFIG_SHA256
        or _digest(P0_CONFIG) != P0_CONFIG_SHA256
    ):
        raise BASE.AuditFailure("P0 recovery execution/config identity changed")
    if (
        len(audit_revision) != 40
        or any(character not in "0123456789abcdef" for character in audit_revision)
        or audit_revision == execution_revision
    ):
        raise BASE.AuditFailure(
            "P0 recovery requires a distinct exact forensic-audit revision"
        )
    if len(verifiers_revision) != 40 or any(
        character not in "0123456789abcdef" for character in verifiers_revision
    ):
        raise BASE.AuditFailure("P0 recovery requires the exact Verifiers revision")

    retained_hashes = {
        relative: _digest(run_dir / relative)
        for relative in P0_RETAINED_ARTIFACT_HASHES
    }
    if retained_hashes != P0_RETAINED_ARTIFACT_HASHES:
        raise BASE.AuditFailure("P0 retained run artifacts changed after execution")
    route_hash = _digest(routing_audit_read_path)
    if route_hash != P0_RETAINED_ROUTING_AUDIT_SHA256:
        raise BASE.AuditFailure("P0 retained route audit changed after execution")

    structural = BASE.validate_runtime(
        run_dir,
        "audit",
        calibration_only=True,
        routing_audit_path=P0_ROUTING_AUDIT,
        routing_audit_read_path=routing_audit_read_path,
        p0_terminal_artifact_recovery=True,
    )
    trace_records = BASE._read_jsonl(
        run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
    )
    route_report = structural["effective_call_routes"]
    if route_report.get("scope") != "retained_P0_postflight_only_not_prospective_admission":
        raise BASE.AuditFailure("P0 recovery did not use the forensic-only route path")
    return {
        "schema_version": "q35-2b-source-route-parity-p0-forensic-recovery/v1",
        "recorded_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "verdict": "posthoc_terminal_artifacts_reconciled",
        "execution_revision": execution_revision,
        "audit_revision": audit_revision,
        "verifiers_revision": verifiers_revision,
        "config": {
            "path": str(P0_CONFIG),
            "sha256": config_sha256,
            "train_seed": BASE.EXPECTED_TRAIN_SEED,
            "instance_offset": BASE.EXPECTED_TRAIN_OFFSET,
        },
        "models": {
            label: {
                "path": str(EXPECTED_MODEL_PATHS[label]),
                "model_sha256": digest,
                "protected_pre_and_post": True,
            }
            for label, digest in EXPECTED_MODEL_HASHES.items()
        },
        "immutable_inputs": {
            "run_dir": str(run_dir),
            "run_artifacts": {
                relative: {
                    "path": str(run_dir / relative),
                    "sha256": digest,
                }
                for relative, digest in retained_hashes.items()
            },
            "routing_audit": {
                "path": str(routing_audit_read_path),
                "sha256": route_hash,
            },
        },
        "reward": _reward_observations(trace_records),
        "termination_and_censoring": structural["prospective_lr0_health"],
        "trainer_inference": structural["prospective_lr0_health"]["metrics"],
        "token_mass": {
            "child_branches": structural["child_branches"],
            "child_trainable_tokens": structural["child_trainable_tokens"],
            "exported_rl_tokens": structural["exported_rl_tokens"],
            "raw_coordinator_sampled_tokens": structural[
                "raw_coordinator_sampled_tokens"
            ],
            "coordinator_exported_trainable_tokens": structural[
                "coordinator_exported_trainable_tokens"
            ],
            "gradient_norm": structural["gradient_norm"],
        },
        "routing_forensics": {
            "forced_assignments": structural["routing"],
            "attached_and_terminal_inventory": route_report,
            "trace_sets": structural["trace_sets"],
            "prospective_exact_multiplicity_pass": False,
            "prospective_failure": "four node=null calls and one successful audit-only route",
        },
        "interpretation": {
            "posthoc_forensic_receipt_only": True,
            "launcher_postflight_passed": False,
            "p0_mechanism_admitted": False,
            "calibration_measurements_recovered": True,
            "numeric_thresholds_evaluated": False,
            "numeric_thresholds_frozen": False,
            "prospective_P1_route_proof_unchanged": True,
            "checkpoint_written": False,
            "optimizer_steps_executed": 1,
            "learning_rate": 0.0,
            "persisted_optimizer_state": False,
            "model_weight_change_expected": False,
            "model_weight_identity_verified_pre_and_post": True,
            "optimizer_update_authorized": False,
            "heldout_evaluation_run": False,
            "next_step_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--execution-revision")
    parser.add_argument("--verifiers-revision")
    parser.add_argument("--config-sha256")
    parser.add_argument("--recover-terminal-artifacts", action="store_true")
    parser.add_argument("--routing-audit-read-path", type=Path)
    parser.add_argument("--audit-revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.recover_terminal_artifacts:
            if (
                not args.runtime
                or args.routing_audit_read_path is None
                or args.audit_revision is None
                or not all(
                    isinstance(value, str) and value
                    for value in (
                        args.execution_revision,
                        args.verifiers_revision,
                        args.config_sha256,
                    )
                )
                or args.output is None
            ):
                raise BASE.AuditFailure(
                    "P0 recovery requires runtime, route path, revisions, config hash, and output"
                )
            if args.output.resolve().is_relative_to(
                args.target.resolve()
            ) or args.output.resolve().is_relative_to(
                args.routing_audit_read_path.resolve().parent
            ):
                raise BASE.AuditFailure(
                    "P0 forensic receipt must be outside the retained artifact tree"
                )
            report = recover_retained_runtime(
                args.target,
                args.routing_audit_read_path,
                execution_revision=args.execution_revision,
                audit_revision=args.audit_revision,
                verifiers_revision=args.verifiers_revision,
                config_sha256=args.config_sha256,
            )
        elif args.runtime:
            if not all(
                isinstance(value, str) and value
                for value in (
                    args.execution_revision,
                    args.verifiers_revision,
                    args.config_sha256,
                )
            ) or args.output is None:
                raise BASE.AuditFailure(
                    "P0 runtime receipt requires revisions, config hash, and output"
                )
            report = validate_runtime(
                args.target,
                execution_revision=args.execution_revision,
                verifiers_revision=args.verifiers_revision,
                config_sha256=args.config_sha256,
            )
        else:
            report = validate_config(args.target)
    except (BASE.AuditFailure, json.JSONDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        with args.output.open("x") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
