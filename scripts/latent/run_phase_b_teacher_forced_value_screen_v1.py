#!/usr/bin/env python3
"""Run the B1 nomination-only teacher-forced local-depth value screen.

The checked-in draft is deliberately non-launchable until evaluator-owned
selection, seed, bank, and threshold fields receive an immutable freeze.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

from prime_rl.phase_b_contract import (
    PhaseBContractError,
    atomic_exclusive_json,
    canonical_json_sha256,
    file_sha256,
    load_json_file,
    normalize_assistant_tool_call_arguments,
)
from prime_rl.phase_b_value_screen import (
    ACTIONS,
    EVALUATION_DEPTHS,
    TRAINING_ARMS,
    TrainingBatch,
    action_margin_from_logits,
    build_action_trie,
    evaluate_nomination,
    midpoint_median,
    paired_loss_deltas,
    validate_evaluation_keys,
    validate_training_batches,
    validate_value_screen_plan,
)

WORKTREE = Path("/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1")
EXPERIMENT = WORKTREE / "experiments/qwen35-2b-latent-coordinator-v1"
DRAFT_PLAN = EXPERIMENT / "phase-b-teacher-forced-value-screen-b1-plan.json"
EXPECTED_ENV = Path("/home/ubuntu/rlm/prime-rl/.venv")
EXPECTED_PYTHONPATH = (
    "/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1/src:"
    "/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1/packages/prime-rl-configs/src"
)
BR5_RUNNER = WORKTREE / "scripts/latent/run_phase_b_fixed_depth_smoke_v1.py"
ARTIFACT_CAP = 536_870_912
MINIMUM_FREE_BYTES = 60 * 1024**3
OUTER_WALL_CLOCK_SECONDS = 14_400
COMPUTE_LIMIT_SECONDS = 14_040
FAILURE_AUDIT_LIMIT_SECONDS = 300
TERMINAL_PUBLICATION_HEADROOM_SECONDS = 60


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError("B1 internal compute wall-clock limit reached")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DRAFT_PLAN)
    parser.add_argument("--train-selection", type=Path, required=True)
    parser.add_argument("--heldout-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--authorized-plan-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _load_br5_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase_b_br5_frozen_runtime", BR5_RUNNER)
    if spec is None or spec.loader is None:
        raise PhaseBContractError("cannot load the frozen BR5 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_br5_receipt(plan: dict[str, Any]) -> dict[str, Any]:
    dependency = plan["br5_dependency"]
    path = Path(dependency["receipt_path"])
    if file_sha256(path) != dependency["receipt_file_sha256"]:
        raise PhaseBContractError("B1 does not bind the exact BR5 SUCCESS file")
    for path_key, hash_key in (
        ("run_log_path", "run_log_sha256"),
        ("preflight_log_path", "preflight_log_sha256"),
        ("snapshot_manifest_path", "snapshot_manifest_sha256"),
    ):
        if file_sha256(Path(dependency[path_key])) != dependency[hash_key]:
            raise PhaseBContractError(f"B1 does not bind exact BR5 evidence: {path_key}")
    receipt = load_json_file(path)
    expected = {
        "status": dependency["required_status"],
        "claim_class": dependency["required_claim_class"],
        "execution_commit": dependency["execution_commit"],
        "plan_sha256": dependency["plan_sha256"],
        "selection_sha256": dependency["selection_sha256"],
        "optimizer": None,
        "optimizer_updates": 0,
        "generation": False,
        "cache": False,
        "worker_loaded": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise PhaseBContractError(f"BR5 SUCCESS predicate differs: {key}")
    bounded = receipt.get("bounded_suffix_equivalence_control")
    breadcrumbs = receipt.get("execution_breadcrumbs")
    if not isinstance(bounded, dict) or not all(
        bounded.get(key) is True for key in ("full_suffix_logits_bitwise_equal", "loss_bitwise_equal")
    ):
        raise PhaseBContractError("BR5 did not prove the exact aligned-suffix control")
    if not isinstance(breadcrumbs, dict) or (
        breadcrumbs.get("completed_feature_captures"),
        breadcrumbs.get("completed_metric_forwards"),
        breadcrumbs.get("completed_backward_arms"),
        breadcrumbs.get("stage"),
    ) != (12, 48, 5, "success_ready"):
        raise PhaseBContractError("BR5 did not complete the frozen connectivity path")
    if receipt.get("maximum_cuda_allocated_bytes") != 27_622_350_848:
        raise PhaseBContractError("BR5 peak allocation predicate differs")
    preflight = receipt.get("tokenizer_preflight_before_model_load")
    if not isinstance(preflight, dict) or (
        preflight.get("completed_rows"),
        preflight.get("model_loaded_during_preflight"),
        preflight.get("cuda_initialized_during_preflight"),
    ) != (12, False, False):
        raise PhaseBContractError("BR5 tokenizer-before-model predicate differs")
    hashes = receipt.get("hashes")
    if not isinstance(hashes, dict) or any(
        hashes.get(before) != hashes.get(after)
        for before, after in (
            ("model_file_pre", "model_file_post"),
            ("model_tensor_tree_pre", "model_tensor_tree_post"),
            ("module_tensor_tree_pre", "module_tensor_tree_post"),
            ("metadata_pre", "metadata_post"),
        )
    ):
        raise PhaseBContractError("BR5 immutable pre/post hashes differ")
    return receipt


def preflight_before_heavy_imports(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_hash = file_sha256(args.plan)
    if plan_hash != args.authorized_plan_sha256:
        raise PhaseBContractError("B1 plan differs from the root-authorized hash")
    plan = load_json_file(args.plan)
    validate_value_screen_plan(plan, require_authorized=True)
    if args.train_selection != Path(plan["training_source"]["selection_path"]):
        raise PhaseBContractError("B1 training order path differs from the frozen plan")
    if args.heldout_selection != Path(plan["heldout"]["selection_path"]):
        raise PhaseBContractError("B1 heldout selection path differs from the frozen plan")
    if file_sha256(args.train_selection) != plan["training_source"]["selection_sha256"]:
        raise PhaseBContractError("B1 training order hash differs from the frozen plan")
    if file_sha256(args.heldout_selection) != plan["heldout"]["selection_sha256"]:
        raise PhaseBContractError("B1 heldout selection hash differs from the frozen plan")
    if args.output_dir != Path(plan["outputs"]["directory"]):
        raise PhaseBContractError("B1 output path differs from the frozen plan")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise PhaseBContractError("B1 requires a fresh output namespace")
    if not args.output_dir.parent.is_dir() or args.output_dir.parent.is_symlink():
        raise PhaseBContractError("B1 output parent is absent or symlinked")
    if shutil.disk_usage(args.output_dir.parent).free < MINIMUM_FREE_BYTES:
        raise PhaseBContractError("B1 host has less than 60 GiB free")
    if len(args.execution_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.execution_commit
    ):
        raise PhaseBContractError("B1 execution commit must be exact lowercase SHA-1")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKTREE, text=True, capture_output=True, check=True
    ).stdout.strip()
    if head != args.execution_commit:
        raise PhaseBContractError("B1 execution commit differs from deployed HEAD")
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=True,
    ).stdout:
        raise PhaseBContractError("B1 requires a clean deployed worktree")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", plan["implementation_commit"], args.execution_commit],
        cwd=WORKTREE,
        check=True,
    )
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve(strict=True):
        raise PhaseBContractError("B1 runner is outside the frozen shared environment")
    if os.environ.get("UV_PROJECT_ENVIRONMENT") != str(EXPECTED_ENV):
        raise PhaseBContractError("B1 UV_PROJECT_ENVIRONMENT differs")
    if os.environ.get("PYTHONPATH") != EXPECTED_PYTHONPATH:
        raise PhaseBContractError("B1 PYTHONPATH differs")
    br5_receipt = _validate_br5_receipt(plan)
    plan["_plan_path"] = str(args.plan)
    plan["_plan_sha256"] = plan_hash
    return (
        plan,
        load_json_file(args.train_selection),
        load_json_file(args.heldout_selection)
        | {
            "_br5_receipt_sha256": file_sha256(Path(plan["br5_dependency"]["receipt_path"])),
            "_br5_receipt": br5_receipt,
        },
    )


def _render_rows(
    rows: list[dict[str, Any]], *, tokenizer: Any, smoke: ModuleType
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    for source in rows:
        task_key = source["task_key"]
        action = source["action"]
        messages = deepcopy(source["messages"])
        source_sha = canonical_json_sha256(source)
        source_messages_sha = canonical_json_sha256(messages)
        normalized, records = normalize_assistant_tool_call_arguments(messages, expected_action=action)
        if canonical_json_sha256(messages) != source_messages_sha or len(records) != 1:
            raise PhaseBContractError(f"B1 row {task_key} normalization escaped its one allowed path")
        reasoning = messages[-1].get("reasoning_content")
        if (
            not isinstance(reasoning, str)
            or not reasoning.strip()
            or normalized[-1].get("reasoning_content") != reasoning
        ):
            raise PhaseBContractError(f"B1 row {task_key} changed or lacks reasoning_content")
        restored = deepcopy(normalized)
        restored[-1]["tool_calls"][0]["function"]["arguments"] = messages[-1]["tool_calls"][0]["function"]["arguments"]
        if restored != messages:
            raise PhaseBContractError(f"B1 row {task_key} changed outside function.arguments")
        raw_tools = source["tools"]
        tools = json.loads(raw_tools) if isinstance(raw_tools, str) else deepcopy(raw_tools)
        plain = smoke._render(tokenizer, normalized[:-1], tools, False)
        opening = smoke._render(tokenizer, normalized[:-1], tools, True)
        full = smoke._render(tokenizer, normalized, tools, False)
        if not opening.startswith(plain) or not full.startswith(opening):
            raise PhaseBContractError(f"B1 row {task_key} violates explicit-thinking string prefixes")
        plain_ids = smoke._token_ids_list(tokenizer, plain)
        open_ids = smoke._token_ids_list(tokenizer, opening)
        full_ids = smoke._token_ids_list(tokenizer, full)
        if open_ids[: len(plain_ids)] != plain_ids or full_ids[: len(open_ids)] != open_ids:
            raise PhaseBContractError(f"B1 row {task_key} violates explicit-thinking token prefixes")
        counterfactual_suffixes: dict[str, list[int]] = {}
        counterfactual_target_hashes: dict[str, str] = {}
        for candidate_action in ACTIONS:
            candidate = deepcopy(normalized)
            candidate[-1]["tool_calls"][0]["function"]["arguments"]["action"] = candidate_action
            candidate_full = smoke._render(tokenizer, candidate, tools, False)
            if not candidate_full.startswith(opening):
                raise PhaseBContractError(f"B1 row {task_key} counterfactual target changed the opening")
            candidate_ids = smoke._token_ids_list(tokenizer, candidate_full)
            if candidate_ids[: len(open_ids)] != open_ids:
                raise PhaseBContractError(f"B1 row {task_key} counterfactual tokens changed the opening")
            counterfactual_suffixes[candidate_action] = candidate_ids[len(open_ids) :]
            counterfactual_target_hashes[candidate_action] = canonical_json_sha256(candidate[-1])
        if counterfactual_suffixes[action] != full_ids[len(open_ids) :]:
            raise PhaseBContractError(f"B1 row {task_key} correct trie leaf differs from its normalized target")
        action_trie = build_action_trie(counterfactual_suffixes, correct_action=action)
        if any(branch["target_offset"] >= len(full_ids) - len(open_ids) for branch in action_trie["branches"]):
            raise PhaseBContractError(f"B1 row {task_key} trie branch is outside the supervised suffix")
        proof = {
            "task_key": task_key,
            "action": action,
            "source_row_sha256": source_sha,
            "reasoning_content_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
            "modified_path": records[0]["modified_path"],
            "plain_ids_sha256": canonical_json_sha256(plain_ids),
            "opening_ids_sha256": canonical_json_sha256(open_ids),
            "full_ids_sha256": canonical_json_sha256(full_ids),
            "plain_tokens": len(plain_ids),
            "opening_tokens": len(open_ids),
            "full_tokens": len(full_ids),
            "counterfactual_target_sha256": counterfactual_target_hashes,
            "action_trie_sha256": action_trie["canonical_trie_sha256"],
            "action_trie_branch_count": action_trie["branch_count"],
        }
        proofs.append(proof)
        prepared.append(
            {
                "task_key": task_key,
                "action": action,
                "plain_ids": plain_ids,
                "open_ids": open_ids,
                "full_ids": full_ids,
                "action_trie": action_trie,
                "render_proof": proof,
            }
        )
    return prepared, proofs


def tokenizer_preflight(
    plan: dict[str, Any],
    train_selection: dict[str, Any],
    heldout_selection: dict[str, Any],
    *,
    parquet: Any,
    AutoTokenizer: Any,
) -> dict[str, Any]:
    smoke = _load_br5_runner()
    smoke._validate_transformers_runtime(plan, transformers=sys.modules["transformers"])
    model_path = Path(plan["protected_model"]["path"])
    if file_sha256(smoke._model_file(model_path)) != plan["protected_model"]["weight_sha256"]:
        raise PhaseBContractError("B1 e33 file hash differs")
    if smoke._metadata_hashes(model_path) != plan["model_metadata_sha256"]:
        raise PhaseBContractError("B1 e33 metadata differs")
    training = plan["training_source"]
    heldout = plan["heldout"]
    for path_key, hash_key, source in (
        ("parquet_path", "parquet_sha256", training),
        ("manifest_path", "manifest_sha256", training),
        ("parquet_path", "parquet_sha256", heldout),
        ("manifest_path", "manifest_sha256", heldout),
    ):
        if file_sha256(Path(source[path_key])) != source[hash_key]:
            raise PhaseBContractError(f"B1 dataset artifact differs: {source[path_key]}")
    for path_key, hash_key in (
        ("generator_path", "generator_sha256"),
        ("runtime_source_path", "runtime_source_sha256"),
        ("taskset_source_path", "taskset_source_sha256"),
    ):
        if file_sha256(Path(heldout[path_key])) != heldout[hash_key]:
            raise PhaseBContractError(f"B1 heldout provenance source differs: {path_key}")
    training_manifest = load_json_file(Path(training["manifest_path"]))
    heldout_manifest = load_json_file(Path(heldout["manifest_path"]))
    if training_manifest.get("training_variants") != [0, 1, 2, 3]:
        raise PhaseBContractError("B1/e33 training manifest variants differ")
    if canonical_json_sha256(training_manifest.get("task_keys")) != plan["training_source"]["ordered_task_key_sha256"]:
        raise PhaseBContractError("B1/e33 training manifest key order differs")
    if heldout_manifest.get("generation", {}).get("template_variants") != [4, 5]:
        raise PhaseBContractError("B1 heldout bank variants differ")
    if heldout_manifest.get("row_list_canonical_sha256") != plan["heldout"]["row_list_canonical_sha256"]:
        raise PhaseBContractError("B1 heldout canonical row-list hash differs")
    if heldout_manifest.get("generator") != {
        "source_path": "scripts/export_phase_b_value_screen_b1_bank_v1.py",
        "source_sha256": heldout["generator_sha256"],
    }:
        raise PhaseBContractError("B1 heldout generator provenance differs")
    if heldout_manifest.get("taskset") != {
        "commit": heldout["taskset_commit"],
        "source_path": ("experiments/qwen35-2b-latent-coordinator-v1/phase-b-b1-bank-v1/taskset-5283a85.py"),
        "source_sha256": heldout["taskset_source_sha256"],
    }:
        raise PhaseBContractError("B1 heldout taskset provenance differs")
    if heldout_manifest.get("artifacts") != {
        "heldout-selection.json": heldout["selection_sha256"],
        "heldout.parquet": heldout["parquet_sha256"],
        "runtime-source.json": heldout["runtime_source_sha256"],
        "taskset-5283a85.py": heldout["taskset_source_sha256"],
        "training-selection.json": training["selection_sha256"],
    }:
        raise PhaseBContractError("B1 heldout bank artifact closure differs")
    runtime_source = load_json_file(Path(heldout["runtime_source_path"]))
    if (
        runtime_source.get("training_root_rows"),
        runtime_source.get("unique_training_root_system_messages"),
        runtime_source.get("root_system_message_sha256"),
        runtime_source.get("reconstruction_byte_exact"),
    ) != (
        32,
        1,
        "59514a723921737f4ad6fcab55c82aee464050297bf343591ea8e2d7950c60b0",
        True,
    ):
        raise PhaseBContractError("B1 runtime-template reconstruction evidence differs")
    training_rows = parquet.read_table(Path(training["parquet_path"])).to_pylist()
    batches = validate_training_batches(training_rows, train_selection)
    by_train_key = {row["task_key"]: row for row in training_rows}
    ordered_training = [by_train_key[key] for batch in batches for key in batch.task_keys]
    heldout_rows = parquet.read_table(Path(heldout["parquet_path"])).to_pylist()
    heldout_keys = validate_evaluation_keys(set(by_train_key), heldout_rows, heldout_selection)
    by_heldout_key = {row["task_key"]: row for row in heldout_rows}
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rendered_train, train_proofs = _render_rows(ordered_training, tokenizer=tokenizer, smoke=smoke)
    rendered_heldout, heldout_proofs = _render_rows(
        [by_heldout_key[key] for key in heldout_keys], tokenizer=tokenizer, smoke=smoke
    )
    if smoke._cuda_initialized_if_torch_loaded():
        raise PhaseBContractError("CUDA initialized during B1 tokenizer preflight")
    return {
        "training_rows": rendered_train,
        "heldout_rows": rendered_heldout,
        "batches": batches,
        "training_proofs": train_proofs,
        "heldout_proofs": heldout_proofs,
        "model_file_sha256": plan["protected_model"]["weight_sha256"],
        "metadata_sha256": plan["model_metadata_sha256"],
    }


def _arm_modules(torch: Any, LocalDepthCodec: Any, OneShotFeedForwardSidecar: Any, TimestepFreeRecurrentSidecar: Any):
    template = LocalDepthCodec().to(device="cuda:0", dtype=torch.bfloat16)
    template_state = {name: value.detach().clone() for name, value in template.state_dict().items()}
    codecs = {arm: LocalDepthCodec().to(device="cuda:0", dtype=torch.bfloat16) for arm in TRAINING_ARMS}
    for codec in codecs.values():
        codec.load_state_dict(template_state, strict=True)
    del template, template_state
    sidecars = {
        "FFN": OneShotFeedForwardSidecar().to(device="cuda:0", dtype=torch.bfloat16),
        "RECURRENT": TimestepFreeRecurrentSidecar().to(device="cuda:0", dtype=torch.bfloat16),
    }
    return codecs, sidecars


def _visible_for_arm(arm: str, anchor: Any, sidecars: dict[str, Any], *, depth: int = 4) -> Any:
    if arm == "STATIC":
        return anchor
    if arm == "FFN":
        return sidecars[arm](anchor)
    return sidecars[arm].rollout(anchor, depth).visible_workspace


def _metric_from_output(output: Any, example: dict[str, Any], *, torch: Any, smoke: ModuleType) -> dict[str, Any]:
    smoke._require_finite(output.loss, "B1 evaluation loss", torch)
    trie = example["action_trie"]
    margin, branches = action_margin_from_logits(
        trie,
        lambda offset, token: float(output.logits[0, offset, token]),
    )
    if max(branch["logit_offset"] for branch in branches) >= output.logits.shape[1] - 1:
        raise PhaseBContractError("B1 action trie branch escaped the aligned supervised suffix")
    return {
        "task_key": example["task_key"],
        "action": example["action"],
        "nll": float(output.loss),
        "margin": margin,
        "branch_count": len(branches),
        "branch_metrics": branches,
        "canonical_trie_sha256": trie["canonical_trie_sha256"],
        "logits_sha256": smoke._tensor_bytes_sha256(output.logits, torch),
    }


def _retention(anchor: Any, visible: Any, *, torch: Any) -> dict[str, float]:
    anchor_flat = anchor.detach().float().reshape(-1)
    visible_flat = visible.detach().float().reshape(-1)
    anchor_norm = torch.linalg.vector_norm(anchor_flat)
    visible_norm = torch.linalg.vector_norm(visible_flat)
    if not bool(torch.isfinite(anchor_norm)) or float(anchor_norm) <= 0.0:
        raise PhaseBContractError("B1 recurrent anchor norm is invalid")
    cosine = torch.nn.functional.cosine_similarity(anchor_flat, visible_flat, dim=0)
    result = {
        "cosine": float(cosine),
        "norm_ratio": float(visible_norm / anchor_norm),
        "relative_l2": float(torch.linalg.vector_norm(visible_flat - anchor_flat) / anchor_norm),
    }
    if not all(torch.isfinite(torch.tensor(value)) for value in result.values()):
        raise PhaseBContractError("B1 recurrent retention metric is non-finite")
    return result


def _stability(diagnostic: Any) -> dict[str, Any]:
    changes = [float(value) for value in diagnostic.memory_change_norms.reshape(-1).tolist()]
    contractions = [float(value) for value in diagnostic.memory_contraction_ratios.reshape(-1).tolist()]
    if len(changes) != 8 or len(contractions) != 7:
        raise PhaseBContractError("B1 T8 diagnostic does not contain eight recurrent updates")
    return {
        "memory_change_rms": changes,
        "memory_contraction_steps_2_8": contractions,
        "median_memory_contraction_steps_2_8": midpoint_median(contractions),
        "max_memory_contraction_steps_2_8": max(contractions),
        "memory_oscillation_rate": float(diagnostic.memory_oscillation_rate),
        "finite": not diagnostic.nonfinite and all(map(torch_isfinite_scalar, (*changes, *contractions))),
    }


def torch_isfinite_scalar(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _evaluate(
    examples: list[dict[str, Any]],
    *,
    model: Any,
    codecs: dict[str, Any],
    sidecars: dict[str, Any],
    shell: Any,
    compose_local_depth_inputs: Any,
    diagnose_recurrent_states: Any,
    torch: Any,
    smoke: ModuleType,
    arms: tuple[str, ...] = ("BASE", "STATIC", "FFN", "RECURRENT"),
) -> dict[str, Any]:
    metrics: dict[str, list[dict[str, Any]]] = {}
    if "BASE" in arms:
        metrics["BASE"] = []
    for arm in ("STATIC", "FFN"):
        if arm in arms:
            metrics[arm] = []
    if "RECURRENT" in arms:
        metrics.update({f"RECURRENT_T{depth}": [] for depth in EVALUATION_DEPTHS})
    with torch.no_grad():
        for example in examples:
            if "BASE" in arms:
                labels, keep, _span = smoke._suffix_loss_arguments(example["labels"])
                base = model(
                    input_ids=example["full_ids"],
                    attention_mask=example["full_mask"],
                    position_ids=example["full_positions"],
                    labels=labels,
                    logits_to_keep=keep,
                    use_cache=False,
                    return_dict=True,
                )
                metrics["BASE"].append(_metric_from_output(base, example, torch=torch, smoke=smoke))
                del base, labels
            hidden = example["captured_hidden"].to("cuda:0")
            mask = torch.ones(hidden.shape[:2], dtype=torch.long, device="cuda:0")
            anchors = {arm: codecs[arm].encode(hidden, mask) for arm in TRAINING_ARMS if arm in arms}
            for arm in ("STATIC", "FFN"):
                if arm not in arms:
                    continue
                visible = _visible_for_arm(arm, anchors[arm], sidecars)
                output = smoke._latent_forward(
                    example,
                    visible,
                    codec=codecs[arm],
                    model=model,
                    embedding_shell_norm=shell,
                    compose_local_depth_inputs=compose_local_depth_inputs,
                    torch=torch,
                )
                metrics[arm].append(_metric_from_output(output, example, torch=torch, smoke=smoke))
                del visible, output
            if "RECURRENT" in arms:
                trajectory = sidecars["RECURRENT"].rollout(anchors["RECURRENT"], 8, return_trajectory=True)
                diagnostic = diagnose_recurrent_states(trajectory)
                if diagnostic.nonfinite:
                    raise PhaseBContractError("B1 recurrent evaluation state is non-finite")
                retention = {
                    f"T{depth}": _retention(anchors["RECURRENT"], trajectory[depth].visible_workspace, torch=torch)
                    for depth in EVALUATION_DEPTHS
                }
                stability = _stability(diagnostic)
                for depth in EVALUATION_DEPTHS:
                    output = smoke._latent_forward(
                        example,
                        trajectory[depth].visible_workspace,
                        codec=codecs["RECURRENT"],
                        model=model,
                        embedding_shell_norm=shell,
                        compose_local_depth_inputs=compose_local_depth_inputs,
                        torch=torch,
                    )
                    metric = _metric_from_output(output, example, torch=torch, smoke=smoke)
                    if depth == 8:
                        metric.update({"retention": retention, "stability_T8": stability})
                    metrics[f"RECURRENT_T{depth}"].append(metric)
                    del output
                del trajectory, diagnostic, retention, stability
            del hidden, mask, anchors
            torch.cuda.empty_cache()
    return metrics


def _train_arm(
    arm: str,
    examples_by_key: dict[str, dict[str, Any]],
    batches: tuple[TrainingBatch, ...],
    *,
    model: Any,
    codec: Any,
    sidecar: Any | None,
    shell: Any,
    compose_local_depth_inputs: Any,
    torch: Any,
    smoke: ModuleType,
    optimizer_plan: dict[str, Any],
    audit_context: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parameters = list(codec.parameters()) + ([] if sidecar is None else list(sidecar.parameters()))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=optimizer_plan["learning_rate"],
        betas=tuple(optimizer_plan["betas"]),
        eps=optimizer_plan["epsilon"],
        weight_decay=optimizer_plan["weight_decay"],
    )
    history: list[dict[str, Any]] = []
    expected_internal_groups = {
        "STATIC": (),
        "FFN": ("ffn_internal",),
        "RECURRENT": ("transition", "memory", "workspace"),
    }[arm]
    internal_gradient_update_passes: dict[str, list[int]] = {group: [] for group in expected_internal_groups}
    for batch in batches:
        audit_context.update({"stage": "training", "arm": arm, "update_index": batch.update_index, "task_key": None})
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for key in batch.task_keys:
            audit_context["task_key"] = key
            example = examples_by_key[key]
            hidden = example["captured_hidden"].to("cuda:0")
            mask = torch.ones(hidden.shape[:2], dtype=torch.long, device="cuda:0")
            anchor = codec.encode(hidden, mask)
            visible = _visible_for_arm(arm, anchor, {} if sidecar is None else {arm: sidecar}, depth=4)
            output = smoke._latent_forward(
                example,
                visible,
                codec=codec,
                model=model,
                embedding_shell_norm=shell,
                compose_local_depth_inputs=compose_local_depth_inputs,
                torch=torch,
            )
            smoke._require_finite(output.loss, f"B1 {arm} training loss", torch)
            losses.append(float(output.loss.detach()))
            (output.loss / len(batch.task_keys)).backward()
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise PhaseBContractError(f"B1 {arm} caused an e33 gradient after row {key}")
            if any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            ):
                raise PhaseBContractError(f"B1 {arm} accumulated a non-finite row gradient")
            del output, visible, anchor, mask, hidden
            torch.cuda.empty_cache()
        named = [(f"codec.{name}", parameter) for name, parameter in codec.named_parameters()]
        if sidecar is not None:
            named.extend((f"sidecar.{name}", parameter) for name, parameter in sidecar.named_parameters())
        gradient_l2 = {
            name: (None if parameter.grad is None else float(torch.linalg.vector_norm(parameter.grad.detach().float())))
            for name, parameter in named
        }
        if any(value is not None and not torch_isfinite_scalar(value) for value in gradient_l2.values()):
            raise PhaseBContractError(f"B1 {arm} has a non-finite named gradient")
        internal_groups: dict[str, float] = {}
        if arm == "FFN":
            internal_groups["ffn_internal"] = math.sqrt(
                math.fsum(
                    value * value
                    for name, value in gradient_l2.items()
                    if name.startswith("sidecar.") and name != "sidecar.output_scale" and value is not None
                )
            )
        elif arm == "RECURRENT":
            for group, prefixes in {
                "transition": ("sidecar.transition.",),
                "memory": ("sidecar.memory_candidate.", "sidecar.memory_gate.", "sidecar.memory_norm."),
                "workspace": ("sidecar.workspace_delta.",),
            }.items():
                internal_groups[group] = math.sqrt(
                    math.fsum(
                        value * value
                        for name, value in gradient_l2.items()
                        if value is not None and any(name.startswith(prefix) for prefix in prefixes)
                    )
                )
        if batch.update_index >= 2:
            for group, value in internal_groups.items():
                if value > 0.0:
                    internal_gradient_update_passes[group].append(batch.update_index)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, optimizer_plan["gradient_clip_norm"])
        if not bool(torch.isfinite(gradient_norm)):
            raise PhaseBContractError(f"B1 {arm} gradient norm is non-finite")
        optimizer.step()
        audit_context["completed_optimizer_updates"][arm] = batch.update_index
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise PhaseBContractError(f"B1 {arm} caused an e33 gradient after update {batch.update_index}")
        output_scale = None if sidecar is None else sidecar.output_scale.detach().float()
        output_scale_evidence = (
            None
            if output_scale is None
            else {
                "finite": bool(torch.isfinite(output_scale).all()),
                "nonzero": bool(torch.count_nonzero(output_scale)),
                "l2": float(torch.linalg.vector_norm(output_scale)),
            }
        )
        if (
            batch.update_index == 1
            and output_scale_evidence is not None
            and not (output_scale_evidence["finite"] and output_scale_evidence["nonzero"])
        ):
            raise PhaseBContractError(f"B1 {arm} output_scale did not open after update one")
        history.append(
            {
                "update_index": batch.update_index,
                "task_keys": list(batch.task_keys),
                "mean_loss": sum(losses) / len(losses),
                "gradient_norm_before_clip": float(gradient_norm),
                "named_gradient_l2": gradient_l2,
                "internal_gradient_groups_l2": internal_groups,
                "output_scale": output_scale_evidence,
                "e33_has_gradient_after_every_row_and_update": False,
            }
        )
        torch.cuda.empty_cache()
    if any(not updates for updates in internal_gradient_update_passes.values()):
        raise PhaseBContractError(f"B1 {arm} lacks a required internal gradient among updates 2..4")
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    audit_context.update({"stage": "optimizer_destroyed", "arm": arm, "task_key": None})
    return history, {
        "internal_gradient_update_passes": internal_gradient_update_passes,
        "optimizer_destroyed_before_next_arm": True,
    }


def _exclusive_torch_save(output: Path, name: str, payload: dict[str, Any], *, torch: Any) -> str:
    final = output / name
    temporary = output / f".{name}.tmp-{os.getpid()}"
    if final.exists() or temporary.exists():
        raise PhaseBContractError(f"B1 compact state target already exists: {name}")
    torch.save(payload, temporary)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.link(temporary, final)
    temporary.unlink()
    return file_sha256(final)


def _cpu_state_dict(module: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _validate_step0_parity(metrics: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    names = ("STATIC", "FFN", *(f"RECURRENT_T{depth}" for depth in EVALUATION_DEPTHS))
    reference = metrics["STATIC"]
    for name in names[1:]:
        if len(metrics[name]) != len(reference):
            raise PhaseBContractError("B1 step-zero parity row counts differ")
        for left, right in zip(reference, metrics[name], strict=True):
            if left["task_key"] != right["task_key"]:
                raise PhaseBContractError("B1 step-zero parity row order differs")
            if (left["logits_sha256"], left["nll"], left["margin"]) != (
                right["logits_sha256"],
                right["nll"],
                right["margin"],
            ):
                raise PhaseBContractError(f"B1 step-zero {name} differs from STATIC")
    return {
        "arms": list(names),
        "rows": len(reference),
        "logits_nll_margin_bitwise_equal": True,
        "reference_sha256": canonical_json_sha256(reference),
    }


def _post_failure_hash_audit(
    plan: dict[str, Any], args: argparse.Namespace, audit_context: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    evidence: dict[str, Any] = {}
    try:
        smoke = _load_br5_runner()
        model_path = Path(plan["protected_model"]["path"])
        evidence["e33_file"] = file_sha256(smoke._model_file(model_path))
        evidence["e33_file_matches"] = evidence["e33_file"] == plan["protected_model"]["weight_sha256"]
        evidence["metadata"] = smoke._metadata_hashes(model_path)
        evidence["metadata_matches"] = evidence["metadata"] == plan["model_metadata_sha256"]
    except BaseException as error:
        errors.append(f"e33_disk: {type(error).__name__}: {error}")
    model = audit_context.get("model")
    if model is not None:
        try:
            current = smoke._module_tensor_sha256(model, audit_context["torch"])
            evidence["e33_tensor_current"] = current
            evidence["e33_tensor_matches_pre"] = current == audit_context.get("e33_tensor_pre")
            evidence["e33_has_gradient"] = any(parameter.grad is not None for parameter in model.parameters())
        except BaseException as error:
            errors.append(f"e33_memory: {type(error).__name__}: {error}")
    modules = audit_context.get("modules")
    if isinstance(modules, dict):
        try:
            evidence["module_tensor_current"] = {
                name: smoke._module_tensor_sha256(module, audit_context["torch"]) for name, module in modules.items()
            }
            evidence["module_tensor_initial"] = audit_context.get("module_pre")
        except BaseException as error:
            errors.append(f"modules: {type(error).__name__}: {error}")
    try:
        evidence["immutable_inputs"] = {
            "plan": file_sha256(args.plan),
            "train_selection": file_sha256(args.train_selection),
            "heldout_selection": file_sha256(args.heldout_selection),
            "training_parquet": file_sha256(Path(plan["training_source"]["parquet_path"])),
            "training_manifest": file_sha256(Path(plan["training_source"]["manifest_path"])),
            "heldout_parquet": file_sha256(Path(plan["heldout"]["parquet_path"])),
            "heldout_manifest": file_sha256(Path(plan["heldout"]["manifest_path"])),
            "heldout_runtime_source": file_sha256(Path(plan["heldout"]["runtime_source_path"])),
            "heldout_generator_source": file_sha256(Path(plan["heldout"]["generator_path"])),
            "heldout_taskset_source": file_sha256(Path(plan["heldout"]["taskset_source_path"])),
            "br5_success": file_sha256(Path(plan["br5_dependency"]["receipt_path"])),
            "br5_run_log": file_sha256(Path(plan["br5_dependency"]["run_log_path"])),
            "br5_preflight_log": file_sha256(Path(plan["br5_dependency"]["preflight_log_path"])),
            "br5_snapshot_manifest": file_sha256(Path(plan["br5_dependency"]["snapshot_manifest_path"])),
        }
    except BaseException as error:
        errors.append(f"immutable_inputs: {type(error).__name__}: {error}")
    checkpoints: dict[str, str] = {}
    for name in ("STATIC.final.pt", "FFN.final.pt", "RECURRENT.final.pt"):
        path = args.output_dir / name
        if path.is_file():
            try:
                checkpoints[name] = file_sha256(path)
            except BaseException as error:
                errors.append(f"checkpoint_{name}: {type(error).__name__}: {error}")
    evidence["compact_checkpoint_hashes"] = checkpoints
    evidence["completed_optimizer_updates"] = deepcopy(audit_context.get("completed_optimizer_updates", {}))
    evidence["execution_breadcrumbs"] = {
        key: audit_context.get(key) for key in ("stage", "arm", "update_index", "task_key")
    }
    torch_module = audit_context.get("torch")
    if torch_module is not None and torch_module.cuda.is_initialized():
        evidence["cuda_memory"] = {
            "allocated_bytes": int(torch_module.cuda.memory_allocated(0)),
            "reserved_bytes": int(torch_module.cuda.memory_reserved(0)),
            "maximum_allocated_bytes": int(torch_module.cuda.max_memory_allocated(0)),
            "maximum_reserved_bytes": int(torch_module.cuda.max_memory_reserved(0)),
        }
    return {"audit_complete": not errors, "hash_probe_error": "; ".join(errors) or None, **evidence}


def execute_value_screen(
    plan: dict[str, Any],
    tokenizer_context: dict[str, Any],
    *,
    execution_commit: str,
    output: Path,
    torch: Any,
    transformers: Any,
    AutoModelForImageTextToText: Any,
    LocalDepthCodec: Any,
    compose_local_depth_inputs: Any,
    OneShotFeedForwardSidecar: Any,
    TimestepFreeRecurrentSidecar: Any,
    diagnose_recurrent_states: Any,
    audit_context: dict[str, Any],
) -> dict[str, Any]:
    smoke = _load_br5_runner()
    smoke._validate_torch_runtime(plan, torch=torch)
    smoke._validate_transformers_runtime(plan, transformers=transformers)
    torch.cuda.set_device(0)
    seed = plan["training"]["initialization_seed"]
    model_path = Path(plan["protected_model"]["path"])
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda:0")
    audit_context.update({"model": model, "model_path": model_path, "torch": torch})
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise PhaseBContractError("B1 protected e33 is not entirely frozen")
    e33_tensor_pre = smoke._module_tensor_sha256(model, torch)
    audit_context["e33_tensor_pre"] = e33_tensor_pre
    e33_file_pre = file_sha256(smoke._model_file(model_path))
    metadata_pre = smoke._metadata_hashes(model_path)
    shell = smoke._mean_embedding_norm(model.get_input_embeddings().weight, torch)
    train_examples = [
        smoke._prepare_example(row, model=model, torch=torch) for row in tokenizer_context["training_rows"]
    ]
    heldout_examples = [
        smoke._prepare_example(row, model=model, torch=torch) for row in tokenizer_context["heldout_rows"]
    ]
    for prepared, rendered in zip(train_examples, tokenizer_context["training_rows"], strict=True):
        prepared["action"] = rendered["action"]
    for prepared, rendered in zip(heldout_examples, tokenizer_context["heldout_rows"], strict=True):
        prepared["action"] = rendered["action"]
        prepared["action_trie"] = rendered["action_trie"]
    feature_hashes_pre = {
        example["task_key"]: smoke._tensor_bytes_sha256(example["captured_hidden"], torch)
        for example in (*train_examples, *heldout_examples)
    }
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng_state_after_seed = {
        "cpu": smoke._tensor_bytes_sha256(torch.get_rng_state(), torch),
        "cuda0": smoke._tensor_bytes_sha256(torch.cuda.get_rng_state(0), torch),
    }
    codecs, sidecars = _arm_modules(torch, LocalDepthCodec, OneShotFeedForwardSidecar, TimestepFreeRecurrentSidecar)
    modules = {
        "STATIC.codec": codecs["STATIC"],
        "FFN.codec": codecs["FFN"],
        "FFN.sidecar": sidecars["FFN"],
        "RECURRENT.codec": codecs["RECURRENT"],
        "RECURRENT.sidecar": sidecars["RECURRENT"],
    }
    module_pre = {name: smoke._module_tensor_sha256(module, torch) for name, module in modules.items()}
    audit_context.update({"modules": modules, "module_pre": module_pre, "completed_optimizer_updates": {}})
    initial_codec_hashes = {module_pre[f"{arm}.codec"] for arm in TRAINING_ARMS}
    if len(initial_codec_hashes) != 1:
        raise PhaseBContractError("B1 codecs are not bitwise-identical at initialization")
    if bool(torch.count_nonzero(sidecars["FFN"].output_scale)) or bool(
        torch.count_nonzero(sidecars["RECURRENT"].output_scale)
    ):
        raise PhaseBContractError("B1 sidecar output scales are not canonically closed")
    rng_state_after_construction = {
        "cpu": smoke._tensor_bytes_sha256(torch.get_rng_state(), torch),
        "cuda0": smoke._tensor_bytes_sha256(torch.cuda.get_rng_state(0), torch),
    }
    heldout_step0 = _evaluate(
        heldout_examples,
        model=model,
        codecs=codecs,
        sidecars=sidecars,
        shell=shell,
        compose_local_depth_inputs=compose_local_depth_inputs,
        diagnose_recurrent_states=diagnose_recurrent_states,
        torch=torch,
        smoke=smoke,
    )
    step0_parity = _validate_step0_parity(heldout_step0)
    audit_context.update({"stage": "baseline_complete", "arm": None, "task_key": None})
    examples_by_key = {example["task_key"]: example for example in train_examples}
    histories: dict[str, Any] = {}
    training_evidence: dict[str, Any] = {}
    heldout_final: dict[str, Any] = {"BASE": deepcopy(heldout_step0["BASE"])}
    optimizer_plan = plan["training"]["optimizer"]
    for arm in TRAINING_ARMS:
        histories[arm], training_evidence[arm] = _train_arm(
            arm,
            examples_by_key,
            tokenizer_context["batches"],
            model=model,
            codec=codecs[arm],
            sidecar=sidecars.get(arm),
            shell=shell,
            compose_local_depth_inputs=compose_local_depth_inputs,
            torch=torch,
            smoke=smoke,
            optimizer_plan=optimizer_plan,
            audit_context=audit_context,
        )
        audit_context["completed_optimizer_updates"][arm] = len(histories[arm])
        arm_metrics = _evaluate(
            heldout_examples,
            model=model,
            codecs=codecs,
            sidecars=sidecars,
            shell=shell,
            compose_local_depth_inputs=compose_local_depth_inputs,
            diagnose_recurrent_states=diagnose_recurrent_states,
            torch=torch,
            smoke=smoke,
            arms=(arm,),
        )
        heldout_final.update(arm_metrics)
        audit_context.update({"stage": "post_arm_evaluation_complete", "arm": arm, "task_key": None})
    module_post = {name: smoke._module_tensor_sha256(module, torch) for name, module in modules.items()}
    if any(module_pre[name] == module_post[name] for name in modules):
        raise PhaseBContractError("B1 expected every trained module tensor tree to change")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise PhaseBContractError("B1 e33 accumulated a gradient")
    e33_tensor_post = smoke._module_tensor_sha256(model, torch)
    e33_file_post = file_sha256(smoke._model_file(model_path))
    metadata_post = smoke._metadata_hashes(model_path)
    if (e33_tensor_pre, e33_file_pre, metadata_pre) != (e33_tensor_post, e33_file_post, metadata_post):
        raise PhaseBContractError("B1 protected e33 changed")
    feature_hashes_post = {
        example["task_key"]: smoke._tensor_bytes_sha256(example["captured_hidden"], torch)
        for example in (*train_examples, *heldout_examples)
    }
    if feature_hashes_pre != feature_hashes_post:
        raise PhaseBContractError("B1 detached host feature cache changed")
    nomination = evaluate_nomination(heldout_step0, heldout_final, safety_gate_passed=True)
    audit_context.update({"stage": "audits_complete", "arm": None, "task_key": None})
    immutable_input_hashes = {
        "plan": file_sha256(Path(plan["_plan_path"])),
        "training_selection": file_sha256(Path(plan["training_source"]["selection_path"])),
        "heldout_selection": file_sha256(Path(plan["heldout"]["selection_path"])),
        "training_parquet": file_sha256(Path(plan["training_source"]["parquet_path"])),
        "training_manifest": file_sha256(Path(plan["training_source"]["manifest_path"])),
        "heldout_parquet": file_sha256(Path(plan["heldout"]["parquet_path"])),
        "heldout_manifest": file_sha256(Path(plan["heldout"]["manifest_path"])),
        "runtime_source": file_sha256(Path(plan["heldout"]["runtime_source_path"])),
        "generator_source": file_sha256(Path(plan["heldout"]["generator_path"])),
        "taskset_source": file_sha256(Path(plan["heldout"]["taskset_source_path"])),
        "br5_success": file_sha256(Path(plan["br5_dependency"]["receipt_path"])),
        "br5_run_log": file_sha256(Path(plan["br5_dependency"]["run_log_path"])),
        "br5_preflight_log": file_sha256(Path(plan["br5_dependency"]["preflight_log_path"])),
        "br5_snapshot_manifest": file_sha256(Path(plan["br5_dependency"]["snapshot_manifest_path"])),
    }
    expected_input_hashes = {
        "plan": plan["_plan_sha256"],
        "training_selection": plan["training_source"]["selection_sha256"],
        "heldout_selection": plan["heldout"]["selection_sha256"],
        "training_parquet": plan["training_source"]["parquet_sha256"],
        "training_manifest": plan["training_source"]["manifest_sha256"],
        "heldout_parquet": plan["heldout"]["parquet_sha256"],
        "heldout_manifest": plan["heldout"]["manifest_sha256"],
        "runtime_source": plan["heldout"]["runtime_source_sha256"],
        "generator_source": plan["heldout"]["generator_sha256"],
        "taskset_source": plan["heldout"]["taskset_source_sha256"],
        "br5_success": plan["br5_dependency"]["receipt_file_sha256"],
        "br5_run_log": plan["br5_dependency"]["run_log_sha256"],
        "br5_preflight_log": plan["br5_dependency"]["preflight_log_sha256"],
        "br5_snapshot_manifest": plan["br5_dependency"]["snapshot_manifest_sha256"],
    }
    if immutable_input_hashes != expected_input_hashes:
        raise PhaseBContractError("B1 immutable input hashes changed during execution")
    paired_contrasts = {
        "RECURRENT_T4_minus_FFN": paired_loss_deltas(heldout_final, "RECURRENT_T4", "FFN"),
        "RECURRENT_T4_minus_RECURRENT_T1": paired_loss_deltas(heldout_final, "RECURRENT_T4", "RECURRENT_T1"),
        "RECURRENT_T4_minus_STATIC": paired_loss_deltas(heldout_final, "RECURRENT_T4", "STATIC"),
        "STATIC_minus_BASE": paired_loss_deltas(heldout_final, "STATIC", "BASE"),
    }
    checkpoint_hashes = {
        "STATIC.final.pt": _exclusive_torch_save(
            output,
            "STATIC.final.pt",
            {"schema_version": "phase-b1-module-state/v1", "arm": "STATIC", "codec": _cpu_state_dict(codecs["STATIC"])},
            torch=torch,
        ),
        "FFN.final.pt": _exclusive_torch_save(
            output,
            "FFN.final.pt",
            {
                "schema_version": "phase-b1-module-state/v1",
                "arm": "FFN",
                "codec": _cpu_state_dict(codecs["FFN"]),
                "sidecar": _cpu_state_dict(sidecars["FFN"]),
            },
            torch=torch,
        ),
        "RECURRENT.final.pt": _exclusive_torch_save(
            output,
            "RECURRENT.final.pt",
            {
                "schema_version": "phase-b1-module-state/v1",
                "arm": "RECURRENT",
                "codec": _cpu_state_dict(codecs["RECURRENT"]),
                "sidecar": _cpu_state_dict(sidecars["RECURRENT"]),
            },
            torch=torch,
        ),
    }
    if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) >= ARTIFACT_CAP:
        raise PhaseBContractError("B1 artifacts reached the 512 MiB cap")
    return {
        "schema_version": "q35-2b-phase-b-teacher-forced-value-screen-success/v1",
        "status": "SUCCESS",
        "disposition": nomination["disposition"],
        "claim_class": plan["claim_class"],
        "execution_commit": execution_commit,
        "optimizer_updates": {arm: len(histories[arm]) for arm in TRAINING_ARMS},
        "row_exposures": {arm: sum(len(item["task_keys"]) for item in histories[arm]) for arm in TRAINING_ARMS},
        "early_stop": False,
        "generation": False,
        "cache": False,
        "worker_loaded": False,
        "strand_a_combined": False,
        "training": histories,
        "training_evidence": training_evidence,
        "evaluation": {
            "step0": heldout_step0,
            "final": heldout_final,
            "depths": list(EVALUATION_DEPTHS),
            "paired_final_loss_contrasts": paired_contrasts,
            "step0_bitwise_parity": step0_parity,
            "nomination": nomination,
        },
        "checkpoint_hashes": checkpoint_hashes,
        "optimizer_state_persisted": False,
        "optimizer_destroyed_before_next_arm": all(
            training_evidence[arm]["optimizer_destroyed_before_next_arm"] for arm in TRAINING_ARMS
        ),
        "initialization": {
            "seed_payload": plan["training"]["initialization_seed_payload"],
            "derivation_sha256": plan["training"]["initialization_derivation_sha256"],
            "seed": seed,
            "construction_order": [
                "codec_template",
                "STATIC.codec",
                "FFN.codec",
                "RECURRENT.codec",
                "FFN.sidecar",
                "RECURRENT.sidecar",
            ],
            "rng_state_after_seed": rng_state_after_seed,
            "rng_state_after_construction": rng_state_after_construction,
            "codec_copies_bitwise_equal": True,
        },
        "hashes": {
            "e33_tensor_pre": e33_tensor_pre,
            "e33_tensor_post": e33_tensor_post,
            "e33_file_pre": e33_file_pre,
            "e33_file_post": e33_file_post,
            "metadata_pre": metadata_pre,
            "metadata_post": metadata_post,
            "modules_pre": module_pre,
            "modules_post": module_post,
            "detached_host_features_pre": feature_hashes_pre,
            "detached_host_features_post": feature_hashes_post,
        },
        "render_proofs": {
            "training": tokenizer_context["training_proofs"],
            "heldout": tokenizer_context["heldout_proofs"],
        },
        "promotion": {
            "admitted": False,
            "b1_nominated": nomination["nominated"],
            "reason": "teacher-forced B1 can nominate only",
            "minimum_complete_live_trajectories_unchanged": 4,
            "teacher_forced_rows_count_as_live_trajectories": False,
            "complete_live_trajectory_count": 0,
        },
        "br5_success_sha256": plan["br5_dependency"]["receipt_file_sha256"],
        "immutable_input_hashes": immutable_input_hashes,
        "maximum_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
    }


def main() -> int:
    args = parse_args()
    started = time.time()
    audit_context: dict[str, Any] = {}
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(COMPUTE_LIMIT_SECONDS)
    try:
        plan, train_selection, heldout_selection = preflight_before_heavy_imports(args)
    except BaseException as error:
        signal.alarm(0)
        print(f"B1 preflight refusal: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    import pyarrow.parquet as parquet
    import transformers
    from transformers import AutoTokenizer

    try:
        context = tokenizer_preflight(
            plan, train_selection, heldout_selection, parquet=parquet, AutoTokenizer=AutoTokenizer
        )
        if args.preflight_only:
            signal.alarm(0)
            print(
                json.dumps(
                    {
                        "status": "B1_TOKENIZER_PREFLIGHT_ONLY_SUCCESS",
                        "training_rows": len(context["training_rows"]),
                        "heldout_rows": len(context["heldout_rows"]),
                        "model_loaded": False,
                        "cuda_initialized": False,
                        "output_created": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        args.output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        import torch
        from transformers import AutoModelForImageTextToText

        from prime_rl.latent.local_depth import LocalDepthCodec, compose_local_depth_inputs
        from prime_rl.latent.recurrent import (
            OneShotFeedForwardSidecar,
            TimestepFreeRecurrentSidecar,
            diagnose_recurrent_states,
        )

        receipt = execute_value_screen(
            plan,
            context,
            execution_commit=args.execution_commit,
            output=args.output_dir,
            torch=torch,
            transformers=transformers,
            AutoModelForImageTextToText=AutoModelForImageTextToText,
            LocalDepthCodec=LocalDepthCodec,
            compose_local_depth_inputs=compose_local_depth_inputs,
            OneShotFeedForwardSidecar=OneShotFeedForwardSidecar,
            TimestepFreeRecurrentSidecar=TimestepFreeRecurrentSidecar,
            diagnose_recurrent_states=diagnose_recurrent_states,
            audit_context=audit_context,
        )
        receipt["elapsed_seconds"] = time.time() - started
        receipt["plan_sha256"] = args.authorized_plan_sha256
        receipt["train_selection_sha256"] = file_sha256(args.train_selection)
        receipt["heldout_selection_sha256"] = file_sha256(args.heldout_selection)
        receipt["wall_clock_contract"] = {
            "outer_seconds": OUTER_WALL_CLOCK_SECONDS,
            "compute_seconds": COMPUTE_LIMIT_SECONDS,
            "failure_audit_seconds": FAILURE_AUDIT_LIMIT_SECONDS,
            "terminal_publication_headroom_seconds": TERMINAL_PUBLICATION_HEADROOM_SECONDS,
        }
        signal.alarm(0)
        atomic_exclusive_json(args.output_dir, "SUCCESS.json", receipt, maximum_directory_bytes=ARTIFACT_CAP)
        return 0
    except BaseException as error:
        if args.output_dir.is_dir():
            signal.alarm(FAILURE_AUDIT_LIMIT_SECONDS)
            post_failure_hash_audit = _post_failure_hash_audit(plan, args, audit_context)
            failure = {
                "schema_version": "q35-2b-phase-b-teacher-forced-value-screen-failure/v1",
                "status": "FAILURE",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.time() - started,
                "plan_sha256": args.authorized_plan_sha256,
                "generation": False,
                "cache": False,
                "worker_loaded": False,
                "strand_a_combined": False,
                "promotion_floor_unchanged": 4,
                "post_failure_hash_audit": post_failure_hash_audit,
                "failure_class": "infrastructure_invalid",
                "wall_clock_contract": {
                    "outer_seconds": OUTER_WALL_CLOCK_SECONDS,
                    "compute_seconds": COMPUTE_LIMIT_SECONDS,
                    "failure_audit_seconds": FAILURE_AUDIT_LIMIT_SECONDS,
                    "terminal_publication_headroom_seconds": TERMINAL_PUBLICATION_HEADROOM_SECONDS,
                },
            }
            try:
                signal.alarm(0)
                atomic_exclusive_json(args.output_dir, "FAILURE.json", failure, maximum_directory_bytes=ARTIFACT_CAP)
            except BaseException as receipt_error:
                print(f"B1 failure receipt publication failed: {receipt_error}", file=sys.stderr)
        print(f"B1 failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
