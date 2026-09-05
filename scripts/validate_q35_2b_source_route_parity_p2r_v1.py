#!/usr/bin/env python3
"""Prospective fail-closed P2R LR=0 route and mechanism admission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "validate_q35_2b_source_first_call_grpo_s6_v1.py"
SPEC = importlib.util.spec_from_file_location("p2_base_validator", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load shared S6 validator")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

CONFIG = Path("experiments/qwen35-2b-document-recursion-zero-update-v1/specialist-source-route-parity-p2r-lr0.toml")
OUTPUT_ROOT = Path("/home/ubuntu/rlm/outputs/q35-2b-source-route-parity-p2r-v1")
RUN_DIR = OUTPUT_ROOT / "lr0-calibration" / "source-route-parity-p2r-lr0-admission"
RESULT_ROOT = Path("/home/ubuntu/rlm/results/q35-2b-source-route-parity-p2r-v1")
ROUTE_AUDIT = RESULT_ROOT / "routing-audit.jsonl"
RESULT = RESULT_ROOT / "mechanism-admission.json"
MODEL_PREFLIGHT = RESULT_ROOT / "preflight-model-hashes.json"
MODEL_POSTFLIGHT = RESULT_ROOT / "postflight-model-hashes.json"
CORE_IMPORT_PROVENANCE = RESULT_ROOT / "core-verifiers-import-provenance.json"
SAMPLING_CONTRACT = Path("experiments/qwen35-2b-document-recursion-zero-update-v1/specialist-source-route-parity-p2r-sampling-v1.json")
FAILED_START_EVIDENCE = Path(
    "experiments/qwen35-2b-document-recursion-zero-update-v1/"
    "specialist-source-route-parity-p2-failed-start-v1.json"
)
FAILED_RUN_DIR = Path(
    "/home/ubuntu/rlm/outputs/q35-2b-source-route-parity-p2-v1/"
    "lr0-calibration/source-route-parity-p2-lr0-admission"
)
FAILED_RESULT_ROOT = Path("/home/ubuntu/rlm/results/q35-2b-source-route-parity-p2-v1")
SEED = 20270925
OFFSET = 72000
MODEL_HASHES = {
    "e33": "e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47",
    "H176": "77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e",
    "S2": "937a96154cd47d8dda1fb01125d9f552037a91bc0542748f417652ceddd47f47",
    "S5": "09a2e3e88030d17896e554211d8fc7eff709d6b4a619e99e3342d05aacde0782",
}
MODEL_PATHS = {
    "e33": str(BASE.E33_PATH),
    "H176": "/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/h176child8-document-child-real12-step8-v2/weights/step_8",
    "S2": "/home/ubuntu/rlm/outputs/q35-2b-specialist-competence-s2-v1/h-source-s2-step8-v1/weights/step_8",
    "S5": str(BASE.S5_PATH),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_core_import_provenance(repo_root: Path | None = None) -> dict[str, object]:
    """Prove the executable imports the reviewed core Verifiers implementation."""

    from verifiers.v1.clients import client as core_client
    from verifiers.v1.clients import train as core_train

    root = (repo_root or Path.cwd()).resolve()
    expected_train = (root / "deps/verifiers/verifiers/v1/clients/train.py").resolve()
    expected_client = (root / "deps/verifiers/verifiers/v1/clients/client.py").resolve()
    actual_train = Path(core_train.__file__).resolve()
    actual_client = Path(core_client.__file__).resolve()
    helper = getattr(core_train, "forwarded_session_headers", None)
    if (
        actual_train != expected_train
        or actual_client != expected_client
        or not callable(helper)
        or getattr(core_client, "SESSION_ID_HEADER", None) != "X-Session-ID"
        or getattr(core_client, "CLIENT_SESSION_ID_HEADER", None)
        != "X-Client-Session-ID"
    ):
        raise BASE.AuditFailure("P2R core Verifiers import provenance changed")
    observed = helper(
        "rollout-trace",
        {"session_id": "prime-agent-branch"},
    )
    if observed != {
        "X-Session-ID": "rollout-trace",
        "X-Client-Session-ID": "prime-agent-branch",
    }:
        raise BASE.AuditFailure("P2R core Verifiers header-forwarding behavior changed")
    return {
        "train_client_module": str(actual_train),
        "train_client_sha256": digest(actual_train),
        "client_module": str(actual_client),
        "client_module_sha256": digest(actual_client),
        "forwarded_session_headers_present": True,
        "rollout_header": "X-Session-ID",
        "branch_header": "X-Client-Session-ID",
        "exact_behavior_probe_passed": True,
    }


def write_core_import_provenance(
    path: Path,
    *,
    execution_revision: str,
    verifiers_revision: str,
    config_sha256: str,
) -> dict[str, object]:
    if (
        path != CORE_IMPORT_PROVENANCE
        or digest(CONFIG) != config_sha256
        or not path.parent.is_dir()
        or not all(
            len(value) == 40
            and all(character in "0123456789abcdef" for character in value)
            for value in (execution_revision, verifiers_revision)
        )
    ):
        raise BASE.AuditFailure("P2R import-provenance receipt identity changed")
    report = {
        "schema_version": "q35-2b-source-route-parity-p2r-core-import-provenance/v1",
        "execution_revision": execution_revision,
        "verifiers_revision": verifiers_revision,
        "config_sha256": config_sha256,
        "provenance": verify_core_import_provenance(),
    }
    with path.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def validate_failed_start_evidence(expected_sha256: str) -> dict[str, object]:
    """Bind same-bank reuse to an immutable, response-free failed P2 start."""

    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or digest(FAILED_START_EVIDENCE) != expected_sha256
    ):
        raise BASE.AuditFailure("P2R failed-start evidence identity changed")
    evidence = BASE._read_json(FAILED_START_EVIDENCE)
    if (
        evidence.get("schema_version")
        != "q35-2b-source-route-parity-p2-failed-start/v1"
        or evidence.get("snapshot_manifest_sha256")
        != "114c0bcbee839e247bb503e33a8bdabb57f0399d398fc48f5a472cfcdf011ab9"
        or evidence.get("execution_revision")
        != "5d71696770808f1f2a08d30d950d5aa9af992e1f"
        or evidence.get("verifiers_revision")
        != "c1a2f5bf3db3f34206e45b04442e64ca6a7770de"
        or evidence.get("failed_run_dir") != str(FAILED_RUN_DIR)
        or evidence.get("failed_result_root") != str(FAILED_RESULT_ROOT)
    ):
        raise BASE.AuditFailure("P2R failed-start manifest metadata changed")
    required = evidence.get("required_artifacts")
    if not isinstance(required, dict) or not required:
        raise BASE.AuditFailure("P2R failed-start artifact inventory is missing")
    actual_hashes = {}
    for relative, expected in required.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise BASE.AuditFailure("P2R failed-start artifact entry is invalid")
        path = (
            FAILED_RESULT_ROOT / relative.removeprefix("result:")
            if relative.startswith("result:")
            else FAILED_RUN_DIR / relative
        )
        if not path.is_file() or digest(path) != expected:
            raise BASE.AuditFailure(f"P2R failed-start artifact changed: {relative}")
        actual_hashes[relative] = expected
    absent = evidence.get("required_absent")
    if not isinstance(absent, list) or not absent:
        raise BASE.AuditFailure("P2R failed-start absence inventory is missing")
    for relative in absent:
        if not isinstance(relative, str):
            raise BASE.AuditFailure("P2R failed-start absence entry is invalid")
        path = (
            FAILED_RESULT_ROOT / relative.removeprefix("result:")
            if relative.startswith("result:")
            else FAILED_RUN_DIR / relative
        )
        if path.exists():
            raise BASE.AuditFailure(f"P2R failed-start artifact unexpectedly exists: {relative}")
    raw = BASE._read_jsonl(
        FAILED_RUN_DIR / "rollouts/step_1/train/all/traces.jsonl"
    )
    calls = [
        call
        for trace in raw
        for call in trace.get("calls", [])
        if isinstance(call, dict)
    ]
    if (
        len(raw) != 8
        or any(trace.get("ok") is not False for trace in raw)
        or len(calls) != 32
        or any(
            call.get("node") is not None
            or not isinstance(call.get("error"), dict)
            or call["error"].get("type") != "ProviderError"
            or call["error"].get("status_code") != 500
            for call in calls
        )
        or (FAILED_RUN_DIR / "metrics.jsonl").stat().st_size != 0
        or any(
            path.is_file()
            for directory in (FAILED_RUN_DIR / "checkpoints", FAILED_RUN_DIR / "weights")
            if directory.exists()
            for path in directory.rglob("*")
        )
    ):
        raise BASE.AuditFailure("P2R failed-start response-free trace proof changed")
    return {
        "evidence_sha256": expected_sha256,
        "snapshot_manifest_sha256": evidence["snapshot_manifest_sha256"],
        "raw_traces": 8,
        "failed_calls": 32,
        "successful_calls": 0,
        "route_events": 0,
        "upstream_model_responses": 0,
        "effective_traces": 0,
        "optimizer_steps": 0,
        "model_artifacts": 0,
        "same_seed_and_bank_reuse_justified": True,
        "artifact_hashes": actual_hashes,
    }


def _actual_model_hashes() -> dict[str, dict[str, object]]:
    rows = {}
    for label, model_path in MODEL_PATHS.items():
        path = Path(model_path)
        if not (path / "STABLE").is_file():
            raise BASE.AuditFailure(f"protected model {label} lacks STABLE")
        actual = digest(path / "model.safetensors")
        if actual != MODEL_HASHES[label]:
            raise BASE.AuditFailure(f"protected model {label} hash changed")
        rows[label] = {"path": model_path, "model_sha256": actual}
    return rows


def write_preflight_model_hashes(path: Path) -> dict[str, object]:
    report = {
        "schema_version": "q35-2b-source-route-parity-p2r-model-preflight/v1",
        "models": _actual_model_hashes(),
    }
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _run_artifact_hashes(run_dir: Path) -> dict[str, str]:
    paths = [
        run_dir / "metrics.jsonl",
        run_dir / "rollouts/step_1/train/all/traces.jsonl",
        run_dir / "rollouts/step_1/train/effective/traces.jsonl",
        run_dir / "configs/orchestrator.json",
        run_dir / "configs/inference.json",
        run_dir / "configs/trainer.json",
        ROUTE_AUDIT,
    ]
    paths.extend(sorted((run_dir / "token_exports/step_1").glob("rank_*.jsonl")))
    if len(paths) != 8 or not all(path.is_file() for path in paths):
        raise BASE.AuditFailure("P2 postflight lacks one complete artifact set")
    return {str(path): digest(path) for path in paths}


def write_postflight_model_hashes(
    path: Path,
    run_dir: Path,
    *,
    execution_revision: str,
    verifiers_revision: str,
    config_sha256: str,
) -> dict[str, object]:
    if path != MODEL_POSTFLIGHT or run_dir != RUN_DIR or digest(CONFIG) != config_sha256:
        raise BASE.AuditFailure("P2 postflight identity changed")
    if not all(
        len(value) == 40 and all(character in "0123456789abcdef" for character in value)
        for value in (execution_revision, verifiers_revision)
    ):
        raise BASE.AuditFailure("P2 postflight requires exact revisions")
    model_files = [
        model_file
        for directory in (run_dir / "checkpoints", run_dir / "weights")
        if directory.exists()
        for model_file in directory.rglob("*")
        if model_file.is_file()
    ]
    if model_files:
        raise BASE.AuditFailure("P2 wrote a forbidden model artifact")
    report = {
        "schema_version": "q35-2b-source-route-parity-p2r-model-postflight/v1",
        "execution_revision": execution_revision,
        "verifiers_revision": verifiers_revision,
        "config_sha256": config_sha256,
        "models": _actual_model_hashes(),
        "run_artifacts": _run_artifact_hashes(run_dir),
        "checkpoint_written": False,
        "learning_rate": 0.0,
    }
    with path.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def validate_config(path: Path) -> dict[str, Any]:
    old_seed, old_offset = BASE.EXPECTED_TRAIN_SEED, BASE.EXPECTED_TRAIN_OFFSET
    try:
        BASE.EXPECTED_TRAIN_SEED, BASE.EXPECTED_TRAIN_OFFSET = SEED, OFFSET
        report = BASE.validate_config(path, "audit")
    finally:
        BASE.EXPECTED_TRAIN_SEED, BASE.EXPECTED_TRAIN_OFFSET = old_seed, old_offset
    payload = tomllib.loads(path.read_text())
    run = payload.get("run", {})
    router = payload.get("inference", {}).get("router", {})
    if path != CONFIG:
        raise BASE.AuditFailure("P2 requires its exact prospective config path")
    if (
        payload.get("output_dir") != str(OUTPUT_ROOT / "lr0-calibration")
        or run.get("name") != RUN_DIR.name
        or run.get("dir") != RUN_DIR.name
        or router.get("state_dir") != str(RESULT_ROOT / "role-router")
        or router.get("audit_log") != str(ROUTE_AUDIT)
        or "eval" in payload
    ):
        raise BASE.AuditFailure("P2 write-once identity or no-eval contract changed")
    sources = payload.get("orchestrator", {}).get("train", {}).get("source", [])
    harnesses = [source.get("env", {}).get("agent", {}).get("harness", {}) for source in sources]
    if (
        router.get("require_client_session_id") is not True
        or len(harnesses) != 2
        or any(
            harness.get("version") != "0.7.2-beta.495.1.97b994c"
            or harness.get("env", {}).get("RLM_MAX_DEPTH") != "1"
            for harness in harnesses
        )
        or any(
            isinstance(section, dict) and "ckpt" in section
            for section in (payload, payload.get("trainer", {}), payload.get("orchestrator", {}))
        )
    ):
        raise BASE.AuditFailure("P2 branch-session or depth-one mechanism changed")
    return {
        **report,
        "seed": SEED,
        "offset": OFFSET,
        "prime_agent_version": "0.7.2-beta.495.1.97b994c",
        "RLM_MAX_DEPTH": "1",
        "depth_semantics": "root depth 0 may spawn terminal depth 1; depth 1 cannot spawn",
        "mechanism_admission_only": True,
    }


def _validate_resolved_p2_mechanism(run_dir: Path) -> dict[str, Any]:
    orchestrator = BASE._read_json(run_dir / "configs/orchestrator.json")
    inference = BASE._read_json(run_dir / "configs/inference.json")
    sources = orchestrator.get("train", {}).get("source", [])
    harnesses = [source.get("env", {}).get("agent", {}).get("harness", {}) for source in sources]
    router = inference.get("router", {})
    if (
        len(harnesses) != 2
        or any(
            harness.get("version") != "0.7.2-beta.495.1.97b994c"
            or harness.get("env", {}).get("RLM_MAX_DEPTH") != "1"
            for harness in harnesses
        )
        or router.get("require_client_session_id") is not True
    ):
        raise BASE.AuditFailure("resolved P2 depth or branch-session mechanism changed")
    return {
        "prime_agent_version": "0.7.2-beta.495.1.97b994c",
        "RLM_MAX_DEPTH": "1",
        "require_client_session_id": True,
        "depth_semantics_verified": True,
    }


def validate_resolved_config_only(run_dir: Path) -> dict[str, Any]:
    BASE._validate_runtime_configs(
        run_dir,
        "audit",
        routing_audit_path=ROUTE_AUDIT,
        expected_train_seed=SEED,
        expected_train_offset=OFFSET,
    )
    return {
        "schema_version": "q35-2b-source-route-parity-p2r-resolved-preflight/v1",
        "run_dir": str(run_dir),
        "mechanism": _validate_resolved_p2_mechanism(run_dir),
    }


def _validate_p2_health(
    structural: dict[str, Any],
    traces: list[dict[str, Any]],
    exports: list[dict[str, Any]],
    bank: dict[str, object],
    observed: dict[str, float],
) -> dict[str, Any]:
    if any(trace.get("ok") is not True or trace.get("errors") for trace in traces):
        raise BASE.AuditFailure("P2 requires all traces ok with no trace errors")
    stops: dict[str, Counter[str]] = defaultdict(Counter)
    rewards: dict[str, list[float]] = defaultdict(list)
    keys: set[str] = set()
    for trace in traces:
        family = trace.get("task", {}).get("data", {}).get("family")
        key = trace.get("task", {}).get("key") or trace.get("task", {}).get("hash")
        reward = trace.get("rewards", {}).get(BASE.EXPECTED_REWARD, {}).get("score")
        if family not in BASE.EXPECTED_FAMILIES or not isinstance(key, str) or not isinstance(reward, (int, float)):
            raise BASE.AuditFailure("P2 trace lacks family/key/reward identity")
        keys.add(key)
        stops[family][trace.get("stop_condition")] += 1
        rewards[family].append(float(reward))
    p2_bank_keys = set(bank["task_keys"])
    p0_bank_keys = set(bank["p0_task_keys"])
    p1_bank_keys = set(bank["p1_task_keys"])
    if (
        not keys <= p2_bank_keys
        or keys & p0_bank_keys
        or keys & p1_bank_keys
        or len(keys) != 2
    ):
        raise BASE.AuditFailure("P2 selected keys are not a disjoint subset of its frozen bank")
    for family in BASE.EXPECTED_FAMILIES:
        if stops[family]["agent_completed"] < 1 or stops[family]["max_turns"] > 7:
            raise BASE.AuditFailure(f"P2 termination gate failed for {family}")
        values = rewards[family]
        mean = statistics.fmean(values)
        if (
            len(set(values)) < 2
            or statistics.pvariance(values) < 0.01
            or max(values) - min(values) < 0.10
            or not any(value > mean for value in values)
            or not any(value < mean for value in values)
        ):
            raise BASE.AuditFailure(f"P2 reward-contrast gate failed for {family}")

    entropy_mean = observed["entropy/all/mean"]
    entropy_max = observed["entropy/all/max"]
    if (
        entropy_mean <= 0
        or entropy_max <= 0
        or observed["is_masked/mean"] > 0.005
    ):
        raise BASE.AuditFailure("P2 aggregate likelihood-parity gate failed")
    active_by_family: Counter[str] = Counter()
    masked_active_by_family: Counter[str] = Counter()
    affected_by_family: Counter[str] = Counter()
    unmasked_mismatch_by_family: dict[str, list[float]] = defaultdict(list)
    unmasked_entropy_by_family: dict[str, list[float]] = defaultdict(list)
    active_abs_mismatch_by_family: dict[str, list[float]] = defaultdict(list)
    active_entropy_by_family: dict[str, list[float]] = defaultdict(list)
    export_rows = []
    for index, export in enumerate(exports):
        source = export.get("env_name")
        mask = export.get("loss_mask")
        masked = export.get("is_masked")
        weights = export.get("rl_weights")
        mismatch = export.get("mismatch_kl")
        entropy = export.get("entropy")
        if source not in BASE.EXPECTED_SOURCE_FAMILIES or not isinstance(mask, list) or not isinstance(masked, list) or not isinstance(mismatch, list) or not isinstance(entropy, list) or len(mask) != len(masked) or len(mask) != len(mismatch) or len(mask) != len(entropy) or (weights is not None and (not isinstance(weights, list) or len(weights) != len(mask))):
            raise BASE.AuditFailure(f"P2 export {index} lacks source/mask evidence")
        active_indices = [
            position
            for position, keep in enumerate(mask)
            if keep
            and (
                weights is None
                or float(1.0 if weights[position] is None else weights[position]) != 0.0
            )
        ]
        if not active_indices:
            raise BASE.AuditFailure("P2 export has no active trainable tokens")
        masked_fraction = sum(bool(masked[position]) for position in active_indices) / len(active_indices)
        if masked_fraction > 0.02:
            raise BASE.AuditFailure("P2 per-export mask fraction exceeds 0.02")
        active_by_family[source] += BASE._active_rl_tokens(export)
        masked_active_by_family[source] += sum(bool(masked[position]) for position in active_indices)
        for position in active_indices:
            mismatch_value = mismatch[position]
            entropy_value = entropy[position]
            if not isinstance(mismatch_value, (int, float)) or not isinstance(entropy_value, (int, float)) or not math.isfinite(mismatch_value) or not math.isfinite(entropy_value):
                raise BASE.AuditFailure("P2 export lacks finite likelihood evidence")
            active_abs_mismatch_by_family[source].append(abs(float(mismatch_value)))
            active_entropy_by_family[source].append(float(entropy_value))
            if not masked[position]:
                unmasked_mismatch_by_family[source].append(float(mismatch_value))
                unmasked_entropy_by_family[source].append(float(entropy_value))
        if masked_fraction > 0:
            affected_by_family[source] += 1
        export_rows.append({"index": index, "source": source, "masked_fraction": masked_fraction})
    if (
        len(exports) != 16
        or any(sum(export.get("env_name") == source for export in exports) != 8 for source in BASE.EXPECTED_SOURCE_FAMILIES)
        or structural["exported_rl_tokens"] < 5000
        or any(active_by_family[source] < 2000 for source in BASE.EXPECTED_SOURCE_FAMILIES)
        or sum(affected_by_family.values()) > 2
        or any(affected_by_family[source] > 1 for source in BASE.EXPECTED_SOURCE_FAMILIES)
        or sum(masked_active_by_family.values()) / sum(active_by_family.values()) > 0.005
    ):
        raise BASE.AuditFailure("P2 export token/masking gate failed")
    for source in BASE.EXPECTED_SOURCE_FAMILIES:
        weighted = masked_active_by_family[source] / active_by_family[source]
        if weighted > 0.01:
            raise BASE.AuditFailure(f"P2 family export masking exceeds 0.01 for {source}")
        family_mismatch_mean = statistics.fmean(unmasked_mismatch_by_family[source])
        family_entropy_mean = statistics.fmean(unmasked_entropy_by_family[source])
        family_entropy_max = max(active_entropy_by_family[source])
        family_mismatch_max_abs = max(active_abs_mismatch_by_family[source])
        if (
            family_entropy_mean <= 0
            or family_entropy_max <= 0
            or abs(family_mismatch_mean) / family_entropy_mean > 0.01
            or family_mismatch_max_abs / family_entropy_max > 0.05
        ):
            raise BASE.AuditFailure(f"P2 export-derived family mismatch gate failed for {source}")
    export_likelihood = {
        source: {
            "unmasked_active_tokens": len(unmasked_mismatch_by_family[source]),
            "unmasked_mismatch_mean": statistics.fmean(unmasked_mismatch_by_family[source]),
            "unmasked_entropy_mean": statistics.fmean(unmasked_entropy_by_family[source]),
            "absolute_unmasked_mismatch_to_entropy_ratio": abs(statistics.fmean(unmasked_mismatch_by_family[source])) / statistics.fmean(unmasked_entropy_by_family[source]),
            "active_mismatch_max_abs": max(active_abs_mismatch_by_family[source]),
            "active_entropy_max": max(active_entropy_by_family[source]),
            "max_abs_mismatch_to_entropy_max_ratio": max(active_abs_mismatch_by_family[source]) / max(active_entropy_by_family[source]),
        }
        for source in BASE.EXPECTED_SOURCE_FAMILIES
    }
    all_unmasked_mismatch = [value for source in BASE.EXPECTED_SOURCE_FAMILIES for value in unmasked_mismatch_by_family[source]]
    all_unmasked_entropy = [value for source in BASE.EXPECTED_SOURCE_FAMILIES for value in unmasked_entropy_by_family[source]]
    all_active_abs_mismatch = [value for source in BASE.EXPECTED_SOURCE_FAMILIES for value in active_abs_mismatch_by_family[source]]
    aggregate_unmasked_mean = statistics.fmean(all_unmasked_mismatch)
    aggregate_unmasked_entropy_mean = statistics.fmean(all_unmasked_entropy)
    if (
        abs(aggregate_unmasked_mean) > 0.003
        or aggregate_unmasked_entropy_mean <= 0
        or abs(aggregate_unmasked_mean) / aggregate_unmasked_entropy_mean > 0.01
        or max(all_active_abs_mismatch) > 0.35
    ):
        raise BASE.AuditFailure("P2 export-derived aggregate mismatch gate failed")
    return {
        "task_keys": sorted(keys),
        "p0_task_keys_disjoint": True,
        "p1_task_keys_disjoint": True,
        "p0_task_key_set_sha256": bank["p0_task_key_set_sha256"],
        "p1_task_key_set_sha256": bank["p1_task_key_set_sha256"],
        "stops": {family: dict(counts) for family, counts in stops.items()},
        "rewards": {family: values for family, values in rewards.items()},
        "active_rl_tokens_by_family": dict(active_by_family),
        "affected_exports_by_family": dict(affected_by_family),
        "masked_active_tokens_by_family": dict(masked_active_by_family),
        "aggregate_export_mask_fraction": sum(masked_active_by_family.values())
        / sum(active_by_family.values()),
        "export_masking": export_rows,
        "likelihood_values": observed,
        "export_derived_family_likelihood": export_likelihood,
        "export_derived_aggregate_likelihood": {
            "unmasked_mismatch_mean": aggregate_unmasked_mean,
            "unmasked_entropy_mean": aggregate_unmasked_entropy_mean,
            "absolute_unmasked_mismatch_to_entropy_ratio": abs(aggregate_unmasked_mean) / aggregate_unmasked_entropy_mean,
            "active_mismatch_max_abs": max(all_active_abs_mismatch),
        },
        "frozen_thresholds_passed": True,
    }


def _materialize_task_bank() -> dict[str, object]:
    hash_path = HERE / "hash_q35_2b_source_route_parity_p2_tasks_v1.py"
    hash_spec = importlib.util.spec_from_file_location("p2_task_hasher", hash_path)
    if hash_spec is None or hash_spec.loader is None:
        raise BASE.AuditFailure("cannot load P2 task-bank materializer")
    hasher = importlib.util.module_from_spec(hash_spec)
    hash_spec.loader.exec_module(hasher)
    return hasher.materialize(CONFIG)


def validate_runtime(
    run_dir: Path,
    *,
    config_sha256: str,
    task_bank_sha256: str,
    task_key_set_sha256: str,
    sampling_contract_sha256: str,
    execution_revision: str,
    verifiers_revision: str,
    preflight_model_hashes: Path,
    postflight_model_hashes: Path,
    core_import_provenance: Path,
    failed_start_evidence_sha256: str,
) -> dict[str, Any]:
    hashes = (config_sha256, task_bank_sha256, task_key_set_sha256, sampling_contract_sha256)
    revisions = (execution_revision, verifiers_revision)
    if (
        run_dir != RUN_DIR
        or preflight_model_hashes != MODEL_PREFLIGHT
        or postflight_model_hashes != MODEL_POSTFLIGHT
        or core_import_provenance != CORE_IMPORT_PROVENANCE
        or digest(CONFIG) != config_sha256
        or digest(SAMPLING_CONTRACT) != sampling_contract_sha256
        or not all(len(value) == 64 and all(character in "0123456789abcdef" for character in value) for value in hashes)
        or not all(len(value) == 40 and all(character in "0123456789abcdef" for character in value) for value in revisions)
    ):
        raise BASE.AuditFailure("P2 runtime/config/task-bank identity changed")
    provenance = BASE._read_json(core_import_provenance)
    if provenance != {
        "schema_version": "q35-2b-source-route-parity-p2r-core-import-provenance/v1",
        "execution_revision": execution_revision,
        "verifiers_revision": verifiers_revision,
        "config_sha256": config_sha256,
        "provenance": verify_core_import_provenance(),
    }:
        raise BASE.AuditFailure("P2R core import-provenance receipt changed")
    failed_start = validate_failed_start_evidence(failed_start_evidence_sha256)
    preflight = BASE._read_json(preflight_model_hashes)
    if (
        preflight.get("schema_version")
        != "q35-2b-source-route-parity-p2r-model-preflight/v1"
        or preflight.get("models") != _actual_model_hashes()
    ):
        raise BASE.AuditFailure("P2 protected-model preflight/postflight evidence differs")
    postflight_models = _actual_model_hashes()
    postflight = BASE._read_json(postflight_model_hashes)
    if (
        postflight.get("schema_version")
        != "q35-2b-source-route-parity-p2r-model-postflight/v1"
        or postflight.get("execution_revision") != execution_revision
        or postflight.get("verifiers_revision") != verifiers_revision
        or postflight.get("config_sha256") != config_sha256
        or postflight.get("models") != postflight_models
        or postflight.get("run_artifacts") != _run_artifact_hashes(run_dir)
        or postflight.get("checkpoint_written") is not False
        or postflight.get("learning_rate") != 0.0
    ):
        raise BASE.AuditFailure("P2 durable postflight protection evidence differs")
    structural = BASE.validate_runtime(
        run_dir,
        "audit",
        routing_audit_path=ROUTE_AUDIT,
        p2_group_atomic_route_admission=True,
        expected_train_seed=SEED,
        expected_train_offset=OFFSET,
    )
    resolved_mechanism = _validate_resolved_p2_mechanism(run_dir)
    traces = BASE._read_jsonl(run_dir / "rollouts/step_1/train/effective/traces.jsonl")
    metrics = BASE._read_jsonl(run_dir / "metrics.jsonl")
    export_dir = run_dir / "token_exports" / "step_1"
    exports = [row for path in sorted(export_dir.glob("rank_*.jsonl")) for row in BASE._read_jsonl(path)]
    bank = _materialize_task_bank()
    if (
        bank["task_bank_sha256"] != task_bank_sha256
        or bank["task_key_set_sha256"] != task_key_set_sha256
        or bank["tasks"] != 64
        or bank["p0_tasks"] != 64
        or bank["p1_tasks"] != 64
        or bank["pairwise_disjoint"] is not True
        or any(bank["overlaps"].values())
    ):
        raise BASE.AuditFailure("P2/P0/P1 full task-bank freeze or disjointness changed")
    observed = BASE._observe_lr0_health(metrics, traces)["metrics"]
    health = _validate_p2_health(structural, traces, exports, bank, observed)
    return {
        "schema_version": "q35-2b-source-route-parity-p2r-mechanism-admission/v1",
        "verdict": "mechanism_admission_pass",
        "config_sha256": config_sha256,
        "task_bank_sha256": task_bank_sha256,
        "task_key_set_sha256": task_key_set_sha256,
        "p0_task_bank_sha256": bank["p0_task_bank_sha256"],
        "p0_task_key_set_sha256": bank["p0_task_key_set_sha256"],
        "p1_task_bank_sha256": bank["p1_task_bank_sha256"],
        "p1_task_key_set_sha256": bank["p1_task_key_set_sha256"],
        "sampling_contract_sha256": sampling_contract_sha256,
        "execution_revision": execution_revision,
        "verifiers_revision": verifiers_revision,
        "core_verifiers_import_provenance": {
            "path": str(core_import_provenance),
            "sha256": digest(core_import_provenance),
            **provenance["provenance"],
        },
        "failed_p2_start": failed_start,
        "protected_models_pre_and_post": {
            "preflight_artifact": {
                "path": str(preflight_model_hashes),
                "sha256": digest(preflight_model_hashes),
                "models": preflight["models"],
            },
            "postflight_recomputed": postflight_models,
            "postflight_artifact": {
                "path": str(postflight_model_hashes),
                "sha256": digest(postflight_model_hashes),
                "models": postflight["models"],
            },
        },
        "artifacts": {
            "metrics_sha256": digest(run_dir / "metrics.jsonl"),
            "raw_traces_sha256": digest(run_dir / "rollouts/step_1/train/all/traces.jsonl"),
            "effective_traces_sha256": digest(run_dir / "rollouts/step_1/train/effective/traces.jsonl"),
            "routing_audit_sha256": digest(ROUTE_AUDIT),
        },
        "routing": structural["effective_call_routes"],
        "raw_effective_partition": structural["trace_sets"],
        "export_partition": {
            "effective_child_branches_matched_one_to_one": True,
            "rejected_groups_have_trainable_advantages": False,
            "rejected_trace_exports": 0,
        },
        "resolved_depth_and_branch_session_mechanism": resolved_mechanism,
        "health": health,
        "gradient_norm": structural["gradient_norm"],
        "checkpoint_written": False,
        "p2_authorizes_nonzero_update": False,
        "heldout_gate_changed": False,
        "downstream_gate": {"hard": 4, "per_family": 2, "recoveries": 4, "regressions": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--config-sha256")
    parser.add_argument("--task-bank-sha256")
    parser.add_argument("--task-key-set-sha256")
    parser.add_argument("--sampling-contract-sha256")
    parser.add_argument("--execution-revision")
    parser.add_argument("--verifiers-revision")
    parser.add_argument("--preflight-model-hashes", type=Path)
    parser.add_argument("--postflight-model-hashes", type=Path)
    parser.add_argument("--core-import-provenance", type=Path)
    parser.add_argument("--failed-start-evidence-sha256")
    parser.add_argument("--write-preflight-model-hashes", type=Path)
    parser.add_argument("--write-postflight-model-hashes", type=Path)
    parser.add_argument("--verify-core-import-provenance", action="store_true")
    parser.add_argument("--write-core-import-provenance", type=Path)
    parser.add_argument("--validate-resolved-configs", action="store_true")
    parser.add_argument("--validate-failed-start-evidence", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.validate_failed_start_evidence:
            report = validate_failed_start_evidence(
                args.failed_start_evidence_sha256 or ""
            )
        elif args.verify_core_import_provenance:
            validate_config(args.target)
            report = verify_core_import_provenance()
        elif args.write_core_import_provenance is not None:
            validate_config(args.target)
            report = write_core_import_provenance(
                args.write_core_import_provenance,
                execution_revision=args.execution_revision or "",
                verifiers_revision=args.verifiers_revision or "",
                config_sha256=args.config_sha256 or "",
            )
        elif args.validate_resolved_configs:
            report = validate_resolved_config_only(args.target)
        elif args.write_preflight_model_hashes is not None:
            validate_config(args.target)
            report = write_preflight_model_hashes(args.write_preflight_model_hashes)
        elif args.write_postflight_model_hashes is not None:
            report = write_postflight_model_hashes(
                args.write_postflight_model_hashes,
                args.target,
                execution_revision=args.execution_revision or "",
                verifiers_revision=args.verifiers_revision or "",
                config_sha256=args.config_sha256 or "",
            )
        elif args.runtime:
            if (
                args.preflight_model_hashes is None
                or args.postflight_model_hashes is None
                or args.core_import_provenance is None
            ):
                raise BASE.AuditFailure(
                    "P2R runtime requires model and core-import preflight evidence"
                )
            report = validate_runtime(
                args.target,
                config_sha256=args.config_sha256 or "",
                task_bank_sha256=args.task_bank_sha256 or "",
                task_key_set_sha256=args.task_key_set_sha256 or "",
                sampling_contract_sha256=args.sampling_contract_sha256 or "",
                execution_revision=args.execution_revision or "",
                verifiers_revision=args.verifiers_revision or "",
                preflight_model_hashes=args.preflight_model_hashes,
                postflight_model_hashes=args.postflight_model_hashes,
                core_import_provenance=args.core_import_provenance,
                failed_start_evidence_sha256=args.failed_start_evidence_sha256 or "",
            )
        else:
            report = validate_config(args.target)
    except (BASE.AuditFailure, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(f"P2 validation failed: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
