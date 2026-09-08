#!/usr/bin/env python3
"""Run a bounded full-dense role update on an audited document bootstrap."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from export_q35_2b_document_decision_sft_v1 import sha256_file

SCHEMA_VERSION = "qwen35-2b-document-decision-update/v1"
DATASET_CONTRACTS = {
    "qwen35-2b-document-decision-sft/v2": (
        "coordinator",
        "canonical_answer_free_first_document_action",
    ),
    "qwen35-2b-document-child-sft/v1": (
        "child",
        "canonical_answer_free_document_leaf_compute_report_stop",
    ),
    "qwen35-2b-document-summary-worker-sft/v1": (
        "child",
        "grounded_english_chapter_summary_report",
    ),
    "qwen35-2b-document-summary-worker-repair-sft/v1": (
        "child",
        "grounded_english_chapter_summary_gate_repair",
    ),
    "qwen35-2b-document-summary-worker-repair-sft/v2": (
        "child",
        "grounded_english_chapter_summary_gate_surgical_repair",
    ),
    "qwen35-2b-document-summary-worker-mixed-sft/v1": (
        "child",
        "grounded_english_chapter_summary_and_gate_repair",
    ),
    "qwen35-2b-document-summary-text-revision-sft/v1": (
        "child",
        "grounded_english_chapter_summary_constrained_revision",
    ),
    "qwen35-2b-document-summary-live-revision-sft/v2": (
        "child",
        "grounded_english_chapter_summary_live_prefix_revision",
    ),
    "qwen35-2b-document-summary-commit-revision-sft/v3": (
        "child",
        "grounded_english_chapter_summary_scaffold_aligned_commit_revision",
    ),
    "qwen35-2b-document-summary-margin-revision-sft/v4": (
        "child",
        "grounded_english_chapter_summary_on_policy_margin_commit_revision",
    ),
    "qwen35-2b-document-coordinator-fanin-sft/v1": (
        "coordinator",
        "grounded_document_coordinator_spawn_partial_yield_fanin",
    ),
    "qwen35-2b-document-coordinator-cleanup-sft/v1": (
        "coordinator",
        "successful_on_policy_document_coordinator_cleanup",
    ),
    "qwen35-2b-document-child-cleanup-sft/v1": (
        "child",
        "successful_on_policy_document_child_cleanup",
    ),
    "qwen35-2b-document-recursive-execution-sft/v1": (
        "coordinator",
        "answer_free_recursive_manager_admission_delegation_and_passive_yield",
    ),
    "qwen35-2b-document-manager-admission-sft/v1": (
        "coordinator",
        "answer_free_depth_two_manager_leaf_admission",
    ),
    "qwen35-2b-document-manager-fanin-sft/v1": (
        "coordinator",
        "depth_two_manager_passive_fanin_parent_report",
    ),
    "qwen35-2b-document-manager-aggregation-sft/v1": (
        "coordinator",
        "depth_two_manager_complete_fanin_parent_report",
    ),
    "qwen35-2b-document-manager-aggregation-permuted-sft/v1": (
        "coordinator",
        "depth_two_manager_order_robust_complete_fanin_parent_report",
    ),
    "qwen35-2b-document-topology-contrast-sft/v1": (
        "coordinator",
        "answer_free_root_worker_versus_manager_topology_choice",
    ),
    "qwen35-2b-document-utility-topology-sft/v1": (
        "coordinator",
        "answer_free_root_topology_choice_from_ownership_and_resource_constraints",
    ),
    "qwen35-2b-document-utility-remedial-sft/v1": (
        "coordinator",
        "answer_free_root_direct_and_hierarchical_utility_remediation",
    ),
    "qwen35-2b-document-hierarchy-remedial-sft/v1": (
        "coordinator",
        "answer_free_root_hierarchical_utility_remediation",
    ),
    "qwen35-2b-document-utility-routed-sft/v1": (
        "coordinator",
        "answer_free_root_topology_choice_from_routed_ownership_and_resource_constraints",
    ),
    "qwen35-2b-document-utility-routed-direct-sft/v1": (
        "coordinator",
        "answer_free_root_direct_utility_choice_from_routed_constraints",
    ),
    "qwen35-2b-document-utility-routed-expanded-sft/v1": (
        "coordinator",
        "answer_free_root_topology_choice_from_expanded_routed_ownership_and_resource_constraints",
    ),
    "qwen35-2b-document-utility-routed-rubric-expanded-sft/v1": (
        "coordinator",
        "answer_free_root_topology_choice_from_routed_resource_decision_rubric",
    ),
    "qwen35-2b-document-utility-routed-consolidated-sft/v2": (
        "coordinator",
        "answer_free_root_topology_choice_from_routed_ownership_and_resource_constraints",
    ),
    "qwen35-2b-document-utility-routed-direct-consolidated-sft/v2": (
        "coordinator",
        "answer_free_root_direct_utility_choice_from_routed_constraints",
    ),
    "qwen35-2b-document-utility-routed-expanded-consolidated-sft/v2": (
        "coordinator",
        "answer_free_root_topology_choice_from_expanded_routed_ownership_and_resource_constraints",
    ),
    "qwen35-2b-document-utility-routed-rubric-expanded-consolidated-sft/v2": (
        "coordinator",
        "answer_free_root_topology_choice_from_routed_resource_decision_rubric",
    ),
    "qwen35-2b-document-utility-routed-causal-matched-consolidated-sft/v1": (
        "coordinator",
        "answer_free_causal_matched_root_topology_choice",
    ),
    "qwen35-2b-document-utility-routed-causal-matched-balanced-consolidated-sft/v1": (
        "coordinator",
        "answer_free_causal_matched_root_topology_choice",
    ),
    "qwen35-2b-adaptive-cognition-sft/v1": (
        "coordinator",
        "answer_free_level_invariant_local_cognition_action",
    ),
    "qwen35-2b-adaptive-cognition-sft/v2": (
        "coordinator",
        "answer_free_level_invariant_local_cognition_action",
    ),
    "qwen35-2b-adaptive-cognition-sft/v3": (
        "coordinator",
        "answer_free_level_invariant_local_cognition_action",
    ),
}
DATASET_ANSWER_FREE = {
    "qwen35-2b-document-decision-sft/v2": True,
    "qwen35-2b-document-child-sft/v1": True,
    "qwen35-2b-document-summary-worker-sft/v1": False,
    "qwen35-2b-document-summary-worker-repair-sft/v1": False,
    "qwen35-2b-document-summary-worker-repair-sft/v2": False,
    "qwen35-2b-document-summary-worker-mixed-sft/v1": False,
    "qwen35-2b-document-summary-text-revision-sft/v1": False,
    "qwen35-2b-document-summary-live-revision-sft/v2": False,
    "qwen35-2b-document-summary-commit-revision-sft/v3": False,
    "qwen35-2b-document-summary-margin-revision-sft/v4": False,
    "qwen35-2b-document-coordinator-fanin-sft/v1": False,
    "qwen35-2b-document-coordinator-cleanup-sft/v1": False,
    "qwen35-2b-document-child-cleanup-sft/v1": True,
    "qwen35-2b-document-recursive-execution-sft/v1": True,
    "qwen35-2b-document-manager-admission-sft/v1": True,
    "qwen35-2b-document-manager-fanin-sft/v1": False,
    "qwen35-2b-document-manager-aggregation-sft/v1": False,
    "qwen35-2b-document-manager-aggregation-permuted-sft/v1": False,
    "qwen35-2b-document-topology-contrast-sft/v1": True,
    "qwen35-2b-document-utility-topology-sft/v1": True,
    "qwen35-2b-document-utility-remedial-sft/v1": True,
    "qwen35-2b-document-hierarchy-remedial-sft/v1": True,
    "qwen35-2b-document-utility-routed-sft/v1": True,
    "qwen35-2b-document-utility-routed-direct-sft/v1": True,
    "qwen35-2b-document-utility-routed-expanded-sft/v1": True,
    "qwen35-2b-document-utility-routed-rubric-expanded-sft/v1": True,
    "qwen35-2b-document-utility-routed-consolidated-sft/v2": True,
    "qwen35-2b-document-utility-routed-direct-consolidated-sft/v2": True,
    "qwen35-2b-document-utility-routed-expanded-consolidated-sft/v2": True,
    "qwen35-2b-document-utility-routed-rubric-expanded-consolidated-sft/v2": True,
    "qwen35-2b-document-utility-routed-causal-matched-consolidated-sft/v1": True,
    "qwen35-2b-document-utility-routed-causal-matched-balanced-consolidated-sft/v1": True,
    "qwen35-2b-adaptive-cognition-sft/v1": True,
    "qwen35-2b-adaptive-cognition-sft/v2": True,
    "qwen35-2b-adaptive-cognition-sft/v3": True,
}
DATASET_ROWS = {schema_version: 12 for schema_version in DATASET_CONTRACTS} | {
    "qwen35-2b-document-summary-worker-mixed-sft/v1": 24,
    "qwen35-2b-document-summary-text-revision-sft/v1": 12,
    "qwen35-2b-document-summary-live-revision-sft/v2": 12,
    "qwen35-2b-document-summary-commit-revision-sft/v3": 12,
    "qwen35-2b-document-summary-margin-revision-sft/v4": 12,
    "qwen35-2b-document-manager-admission-sft/v1": 4,
    "qwen35-2b-document-manager-aggregation-sft/v1": 4,
    "qwen35-2b-document-manager-aggregation-permuted-sft/v1": 24,
    "qwen35-2b-document-topology-contrast-sft/v1": 8,
    "qwen35-2b-document-utility-topology-sft/v1": 6,
    "qwen35-2b-document-utility-remedial-sft/v1": 8,
    "qwen35-2b-document-hierarchy-remedial-sft/v1": 8,
    "qwen35-2b-document-utility-routed-sft/v1": 24,
    "qwen35-2b-document-utility-routed-direct-sft/v1": 8,
    "qwen35-2b-document-utility-routed-expanded-sft/v1": 36,
    "qwen35-2b-document-utility-routed-rubric-expanded-sft/v1": 36,
    "qwen35-2b-document-utility-routed-consolidated-sft/v2": 24,
    "qwen35-2b-document-utility-routed-direct-consolidated-sft/v2": 8,
    "qwen35-2b-document-utility-routed-expanded-consolidated-sft/v2": 36,
    "qwen35-2b-document-utility-routed-rubric-expanded-consolidated-sft/v2": 36,
    "qwen35-2b-document-utility-routed-causal-matched-consolidated-sft/v1": 48,
    "qwen35-2b-document-utility-routed-causal-matched-balanced-consolidated-sft/v1": 36,
    "qwen35-2b-adaptive-cognition-sft/v1": 48,
    "qwen35-2b-adaptive-cognition-sft/v2": 48,
    "qwen35-2b-adaptive-cognition-sft/v3": 48,
}
DATASET_BATCH_SIZES = {
    "qwen35-2b-document-summary-worker-mixed-sft/v1": 12,
    "qwen35-2b-document-summary-text-revision-sft/v1": 12,
    "qwen35-2b-document-summary-live-revision-sft/v2": 12,
    "qwen35-2b-document-summary-commit-revision-sft/v3": 12,
    "qwen35-2b-document-manager-aggregation-permuted-sft/v1": 12,
    "qwen35-2b-document-topology-contrast-sft/v1": 8,
    "qwen35-2b-document-utility-topology-sft/v1": 6,
    "qwen35-2b-document-utility-remedial-sft/v1": 8,
    "qwen35-2b-document-hierarchy-remedial-sft/v1": 8,
    "qwen35-2b-document-utility-routed-sft/v1": 12,
    "qwen35-2b-document-utility-routed-direct-sft/v1": 8,
    "qwen35-2b-document-utility-routed-expanded-sft/v1": 12,
    "qwen35-2b-document-utility-routed-rubric-expanded-sft/v1": 12,
    "qwen35-2b-document-utility-routed-consolidated-sft/v2": 12,
    "qwen35-2b-document-utility-routed-direct-consolidated-sft/v2": 8,
    "qwen35-2b-document-utility-routed-expanded-consolidated-sft/v2": 12,
    "qwen35-2b-document-utility-routed-rubric-expanded-consolidated-sft/v2": 12,
    "qwen35-2b-document-utility-routed-causal-matched-consolidated-sft/v1": 12,
    "qwen35-2b-document-utility-routed-causal-matched-balanced-consolidated-sft/v1": 12,
    "qwen35-2b-adaptive-cognition-sft/v1": 16,
    "qwen35-2b-adaptive-cognition-sft/v2": 16,
    "qwen35-2b-adaptive-cognition-sft/v3": 12,
}
PROMOTION_MINIMUM = 4


def _quote(value: str | Path) -> str:
    return json.dumps(str(value))


def training_config(
    *,
    run_name: str,
    model_path: Path,
    dataset_dir: Path,
    output_root: Path,
    learning_rate: float,
    optimizer_updates: int = 1,
    batch_size: int = 12,
    enable_thinking: bool = False,
) -> str:
    if not 1 <= optimizer_updates <= 8:
        raise ValueError("document decision bootstrap requires one to eight updates")
    if batch_size not in {4, 6, 8, 12, 16}:
        raise ValueError(
            "document decision bootstrap batch size must be 4, 6, 8, 12, or 16"
        )
    return f"""max_steps = {optimizer_updates}
output_dir = {_quote(output_root)}
clean = false

[run]
name = {_quote(run_name)}
dir = {_quote(run_name)}

[deployment]
type = "single_node"
gpus_per_node = 2
num_gpus = 2

[model]
name = {_quote(model_path)}
impl = "custom"
optimization_dtype = "bfloat16"
reduce_dtype = "bfloat16"

[model.vlm]
vision_encoder_attr = "model.visual"
language_model_attr = "model.language_model"
freeze_vision_encoder = true

[model.compile]
fullgraph = false

[model.ac]
mode = "full"
freq = 1
targets = ["norm"]

[tokenizer]
name = {_quote(model_path)}

[renderer]
name = "qwen3.5"
enable_thinking = {str(enable_thinking).lower()}

[data]
type = "sft"
name = {_quote(dataset_dir)}
batch_size = {batch_size}
micro_batch_size = 1
seq_len = 16384
shuffle = false
seed = 20260830

[data.loss_mask]
system = false
user = false
assistant = true
tool = false

[optim]
type = "adamw"
lr = {learning_rate:.12g}
weight_decay = 0.01
max_norm = 1.0
betas1 = 0.9
betas2 = 0.999

[scheduler]
type = "constant"

[ckpt]
interval = 1
keep_last = 1
weights_only = true

[ckpt.weights]
save_sharded = true
save_format = "safetensors"

[file_monitor]
filename = "metrics.jsonl"
"""


def _write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _valid_margin_revision_manifest(manifest: dict[str, Any]) -> bool:
    records = manifest.get("case_records")
    if not isinstance(records, list) or len(records) != 12:
        return False
    expected_targets: dict[str, int] = {}
    repetitions: dict[str, set[int]] = {}
    for record in records:
        if not isinstance(record, dict):
            return False
        chapter_id = record.get("chapter_id")
        repetition = record.get("repetition")
        word_budget = record.get("word_budget")
        if (
            chapter_id not in {"scope", "operations", "exceptions"}
            or not isinstance(repetition, int)
            or not isinstance(word_budget, int)
        ):
            return False
        target = max(0, word_budget - 3)
        if chapter_id in expected_targets and expected_targets[chapter_id] != target:
            return False
        expected_targets[chapter_id] = target
        repetitions.setdefault(chapter_id, set()).add(repetition)
    source_sha = manifest.get("on_policy_source_model_sha256")
    return (
        repetitions == {chapter_id: set(range(4)) for chapter_id in expected_targets}
        and set(expected_targets) == {"scope", "operations", "exceptions"}
        and manifest.get("feedback_target_words_by_chapter") == expected_targets
        and isinstance(manifest.get("on_policy_source_model"), str)
        and isinstance(source_sha, str)
        and len(source_sha) == 64
        and all(character in "0123456789abcdef" for character in source_sha)
    )


def _validated_dataset(path: Path) -> dict[str, Any]:
    manifest_path = path / "MANIFEST.json"
    parquet = path / "train.parquet"
    manifest = json.loads(manifest_path.read_text())
    schema_version = manifest.get("schema_version")
    contract = DATASET_CONTRACTS.get(schema_version)
    expected_family_count = {
        "qwen35-2b-document-utility-topology-sft/v1": 2,
        "qwen35-2b-document-hierarchy-remedial-sft/v1": 8,
        "qwen35-2b-document-utility-routed-sft/v1": 8,
        "qwen35-2b-document-utility-routed-direct-sft/v1": 8,
        "qwen35-2b-document-utility-routed-expanded-sft/v1": 12,
        "qwen35-2b-document-utility-routed-rubric-expanded-sft/v1": 12,
        "qwen35-2b-document-utility-routed-consolidated-sft/v2": 8,
        "qwen35-2b-document-utility-routed-direct-consolidated-sft/v2": 8,
        "qwen35-2b-document-utility-routed-expanded-consolidated-sft/v2": 12,
        "qwen35-2b-document-utility-routed-rubric-expanded-consolidated-sft/v2": 12,
        "qwen35-2b-document-utility-routed-causal-matched-balanced-consolidated-sft/v1": 12,
        "qwen35-2b-adaptive-cognition-sft/v1": 16,
        "qwen35-2b-adaptive-cognition-sft/v2": 16,
        "qwen35-2b-adaptive-cognition-sft/v3": 16,
    }.get(schema_version, 4)
    family_counts = manifest.get("family_counts", {})
    family_counts_valid = set(family_counts.values()) == {expected_family_count}
    if schema_version == (
        "qwen35-2b-document-utility-routed-causal-matched-consolidated-sft/v1"
    ):
        family_counts_valid = family_counts == {
            "document_utility_direct": 12,
            "document_utility_flat": 24,
            "document_utility_hierarchical": 12,
        }
    if (
        contract is None
        or manifest.get("status") != "complete"
        or (manifest.get("role"), manifest.get("objective")) != contract
        or manifest.get("rows") != DATASET_ROWS.get(schema_version)
        or not family_counts_valid
        or (
            schema_version
            in {
                "qwen35-2b-document-summary-live-revision-sft/v2",
                "qwen35-2b-document-summary-commit-revision-sft/v3",
                "qwen35-2b-document-summary-margin-revision-sft/v4",
            }
            and (
                manifest.get("live_revision_role_sequence")
                != [
                    "runtime_user",
                    "task_user",
                    "assistant_draft_context",
                    "revision_feedback_user",
                    "assistant_corrected_target",
                ]
                or manifest.get("assistant_messages_per_row") != 2
                or manifest.get("assistant_target_messages_per_row") != 1
                or manifest.get("context_assistant_messages_per_row") != 1
                or manifest.get("prior_assistant_draft_trainable") is not False
                or manifest.get("corrected_assistant_target_trainable") is not True
                or manifest.get("generation_prompt_common_prefix_trainable")
                is not False
                or manifest.get("completion_boundary_alignment")
                != "renderer_common_prefix_v1"
                or manifest.get("live_transfer_check_required") is not True
                or manifest.get("distinct_conversation_payloads") != 3
                or manifest.get("repetitions_per_chapter") != 4
                or manifest.get("broad_skill_claim") is not False
                or manifest.get("requires_renderer_boundary_audit_before_training")
                is not True
                or (
                    schema_version
                    in {
                        "qwen35-2b-document-summary-commit-revision-sft/v3",
                        "qwen35-2b-document-summary-margin-revision-sft/v4",
                    }
                    and (
                        manifest.get("renderer_enable_thinking") is not False
                        or manifest.get("observed_live_draft_context") is not True
                        or manifest.get("prior_assistant_reasoning_present_in_training")
                        is not False
                        or manifest.get(
                            "live_prior_assistant_reasoning_stripped_by_scaffold"
                        )
                        is not True
                    )
                )
                or (
                    schema_version
                    == "qwen35-2b-document-summary-margin-revision-sft/v4"
                    and (
                        manifest.get("feedback_contract")
                        != "commit_once_with_three_word_safety_margin_v1"
                        or manifest.get("feedback_safety_margin_words") != 3
                        or manifest.get(
                            "observed_current_revision_outputs_trainable"
                        )
                        is not False
                        or manifest.get("on_policy_development_failures_only")
                        is not True
                        or not _valid_margin_revision_manifest(manifest)
                    )
                )
            )
        )
        or (
            schema_version
            in {
                "qwen35-2b-adaptive-cognition-sft/v1",
                "qwen35-2b-adaptive-cognition-sft/v2",
                "qwen35-2b-adaptive-cognition-sft/v3",
            }
            and (
                manifest.get("model_visible_topology_labels") is not False
                or manifest.get("level_invariant_action_contract") is not True
                or manifest.get("root_and_nonroot_coordinator_rows") is not True
                or (
                    schema_version
                    in {
                        "qwen35-2b-adaptive-cognition-sft/v2",
                        "qwen35-2b-adaptive-cognition-sft/v3",
                    }
                    and manifest.get("action_only_tool_arguments") is not True
                )
                or (
                    schema_version == "qwen35-2b-adaptive-cognition-sft/v3"
                    and (
                        manifest.get("training_batch_size") != 12
                        or manifest.get("curriculum_order")
                        != "nonroot_delegation_first"
                    )
                )
                or manifest.get("action_counts")
                != {
                    "solve_owned": 16,
                    "delegate_terminal": 16,
                    "delegate_coordinator": 16,
                }
            )
        )
        or (
            schema_version
            in {
                "qwen35-2b-document-utility-routed-sft/v1",
                "qwen35-2b-document-utility-routed-direct-sft/v1",
                "qwen35-2b-document-utility-routed-expanded-sft/v1",
                "qwen35-2b-document-utility-routed-rubric-expanded-sft/v1",
                "qwen35-2b-document-utility-routed-consolidated-sft/v2",
                "qwen35-2b-document-utility-routed-direct-consolidated-sft/v2",
                "qwen35-2b-document-utility-routed-expanded-consolidated-sft/v2",
                "qwen35-2b-document-utility-routed-rubric-expanded-consolidated-sft/v2",
                "qwen35-2b-document-utility-routed-causal-matched-consolidated-sft/v1",
                "qwen35-2b-document-utility-routed-causal-matched-balanced-consolidated-sft/v1",
            }
            and manifest.get("root_coordinator_contract_aligned") is not True
        )
        or (
            schema_version
            in {
                "qwen35-2b-document-utility-routed-rubric-expanded-sft/v1",
                "qwen35-2b-document-utility-routed-rubric-expanded-consolidated-sft/v2",
                "qwen35-2b-document-utility-routed-causal-matched-consolidated-sft/v1",
                "qwen35-2b-document-utility-routed-causal-matched-balanced-consolidated-sft/v1",
            }
            and (
                manifest.get("utility_decision_rubric_aligned") is not True
                or manifest.get("leading_system_message_count") != 1
                or manifest.get("utility_decision_rubric_serialization")
                not in {
                    "renderer_compatible_merged_system_v1",
                    "vllm_developer_consolidated_system_v1",
                }
            )
        )
        or (
            schema_version
            in {
                "qwen35-2b-document-utility-routed-consolidated-sft/v2",
                "qwen35-2b-document-utility-routed-direct-consolidated-sft/v2",
                "qwen35-2b-document-utility-routed-expanded-consolidated-sft/v2",
                "qwen35-2b-document-utility-routed-rubric-expanded-consolidated-sft/v2",
                "qwen35-2b-document-utility-routed-causal-matched-consolidated-sft/v1",
                "qwen35-2b-document-utility-routed-causal-matched-balanced-consolidated-sft/v1",
            }
            and (
                manifest.get("vllm_developer_system_consolidation_aligned")
                is not True
                or manifest.get("prime_agent_instruction_consolidated") is not True
                or not isinstance(manifest.get("live_system_prefix_count"), int)
                or not 1
                <= manifest["live_system_prefix_count"]
                <= manifest.get("rows", 0)
                or not isinstance(
                    manifest.get("live_system_prefix_set_sha256"), str
                )
            )
        )
        or manifest.get("answer_free") is not DATASET_ANSWER_FREE.get(schema_version)
        or manifest.get("tool_call_format") != "openai_function_v1"
        or manifest.get("dataset", {}).get("path") != parquet.name
        or manifest.get("dataset", {}).get("sha256") != sha256_file(parquet)
    ):
        raise ValueError(f"invalid document decision bootstrap: {path}")
    return manifest


def _metrics(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.update(json.loads(line))
    required = ("loss/mean", "loss/nan_count", "optim/grad_norm", "time/step")
    if any(key not in result for key in required) or result["loss/nan_count"] != 0:
        raise ValueError(f"incomplete document decision metrics: {path}")
    for key in ("loss/mean", "optim/grad_norm", "time/step"):
        if not isinstance(result[key], int | float) or not math.isfinite(result[key]):
            raise ValueError(f"non-finite document decision metric {key}: {result[key]!r}")
    return result


def _validated_renderer_audit(path: Path, dataset: dict[str, Any]) -> dict[str, Any]:
    audit_path = path / "RENDERER-AUDIT.json"
    if not audit_path.is_file():
        raise ValueError(f"missing live revision renderer audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    schema_version = dataset["schema_version"]
    commit_revision = schema_version in {
        "qwen35-2b-document-summary-commit-revision-sft/v3",
        "qwen35-2b-document-summary-margin-revision-sft/v4",
    }
    expected_mode = not commit_revision
    records = audit.get("chapters") if commit_revision else audit.get("modes")
    selected = next(
        (
            record
            for record in records or []
            if isinstance(record, dict)
            and record.get("enable_thinking") is expected_mode
        ),
        None,
    )
    expected_audit_schema = {
        "qwen35-2b-document-summary-commit-revision-sft/v3": (
            "qwen35-2b-document-summary-commit-revision-renderer-audit/v3"
        ),
        "qwen35-2b-document-summary-margin-revision-sft/v4": (
            "qwen35-2b-document-summary-margin-revision-renderer-audit/v4"
        ),
    }.get(
        schema_version,
        "qwen35-2b-document-summary-live-revision-renderer-audit/v2",
    )
    exact_history = (
        len(records or []) == 3
        and [record.get("chapter_id") for record in records]
        == ["scope", "operations", "exceptions"]
        and all(
            record.get("reasoning_stripped_history_token_equivalent") is True
            and record.get("generation_prompt_is_full_prefix") is True
            and record.get("prior_assistant_trainable_tokens") == 0
            and record.get("truncated") is False
            and record.get("completion_boundary_alignment")
            == "renderer_common_prefix_v1"
            and isinstance(record.get("trainable_completion_tokens"), int)
            and record["trainable_completion_tokens"] > 0
            for record in records or []
        )
        if commit_revision
        else selected is not None
        and selected.get("live_history_token_equivalent") is True
    )
    if (
        audit.get("schema_version") != expected_audit_schema
        or audit.get("status") != "complete"
        or audit.get("dataset_schema_version") != dataset["schema_version"]
        or audit.get("dataset_manifest_sha256")
        != sha256_file(path / "MANIFEST.json")
        or audit.get("dataset_parquet_sha256") != sha256_file(path / "train.parquet")
        or audit.get("selected_enable_thinking") is not expected_mode
        or (
            schema_version == "qwen35-2b-document-summary-margin-revision-sft/v4"
            and (
                audit.get("feedback_safety_margin_words") != 3
                or audit.get("on_policy_margin_prefix_verified") is not True
            )
        )
        or audit.get("cuda_initialized") is not False
        or (
            not commit_revision
            and (
                audit.get("live_task_prompt_equal") is not True
                or audit.get("live_prior_draft_equal") is not True
                or audit.get("live_revision_feedback_equal") is not True
            )
        )
        or audit.get("role_sequence")
        != ["user", "user", "assistant", "user", "assistant"]
        or selected is None
        or not exact_history
        or selected.get("prior_assistant_trainable_tokens") != 0
        or selected.get("truncated") is not False
        or selected.get("completion_boundary_alignment")
        != "renderer_common_prefix_v1"
        or not isinstance(selected.get("trainable_completion_tokens"), int)
        or selected["trainable_completion_tokens"] <= 0
    ):
        raise ValueError(f"invalid live revision renderer audit: {audit_path}")
    return audit


def _gpus_idle() -> bool:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_model = args.source_model.resolve()
    source_weight = source_model / "model.safetensors"
    if not source_weight.is_file():
        raise FileNotFoundError(source_weight)
    if (source_model / "STABLE").exists() is False:
        raise ValueError(f"source checkpoint is not marked stable: {source_model}")
    source_sha = sha256_file(source_weight)
    dataset_dir = args.dataset_dir.resolve()
    dataset = _validated_dataset(dataset_dir)
    if (
        dataset["schema_version"]
        == "qwen35-2b-document-summary-margin-revision-sft/v4"
        and dataset["on_policy_source_model_sha256"] != source_sha
    ):
        raise ValueError("margin revision dataset does not match its source checkpoint")
    renderer_audit = None
    if dataset["schema_version"] in {
        "qwen35-2b-document-summary-live-revision-sft/v2",
        "qwen35-2b-document-summary-commit-revision-sft/v3",
        "qwen35-2b-document-summary-margin-revision-sft/v4",
    }:
        renderer_audit = _validated_renderer_audit(dataset_dir, dataset)
    output_root = args.output_root.resolve()
    state_dir = args.state_dir.resolve()
    config_path = state_dir / f"{args.run_name}.toml"
    config = training_config(
        run_name=args.run_name,
        model_path=source_model,
        dataset_dir=dataset_dir,
        output_root=output_root,
        learning_rate=args.learning_rate,
        optimizer_updates=args.optimizer_updates,
        batch_size=DATASET_BATCH_SIZES.get(dataset["schema_version"], dataset["rows"]),
        enable_thinking=args.enable_thinking,
    )
    _write_once(config_path, config)

    output_dir = output_root / args.run_name
    output_model = output_dir / f"weights/step_{args.optimizer_updates}"
    output_weight = output_model / "model.safetensors"
    metrics_path = output_dir / "metrics.jsonl"
    complete = output_weight.is_file() and (output_model / "STABLE").is_file() and metrics_path.is_file()
    if not complete:
        if not _gpus_idle():
            raise RuntimeError("GPUs are not idle at the document decision update boundary")
        env = os.environ.copy()
        env.update({"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "0"})
        subprocess.run(
            [str(args.uv_bin), "run", "--no-sync", "sft", "@", str(config_path)],
            cwd=args.repo.resolve(),
            env=env,
            check=True,
            timeout=args.timeout,
        )
    metrics = _metrics(metrics_path)
    output_sha = sha256_file(output_weight)
    if output_sha == source_sha:
        raise ValueError("document decision update did not change the dense weights")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "algorithm": f"sft_document_{dataset['role']}_bootstrap_v1",
        "role": dataset["role"],
        "optimizer_updates": args.optimizer_updates,
        "full_dense": True,
        "promotion_minimum": PROMOTION_MINIMUM,
        "source": {"model_path": str(source_model), "model_sha256": source_sha},
        "dataset": {
            "path": str(dataset_dir),
            "manifest_sha256": sha256_file(dataset_dir / "MANIFEST.json"),
            "train_parquet_sha256": sha256_file(dataset_dir / "train.parquet"),
            "rows": dataset["rows"],
            "family_counts": dataset["family_counts"],
            "answer_free": dataset["answer_free"],
            "tool_call_format": dataset["tool_call_format"],
            "renderer_audit_sha256": (
                None
                if renderer_audit is None
                else sha256_file(dataset_dir / "RENDERER-AUDIT.json")
            ),
        },
        "training": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "learning_rate": args.learning_rate,
            "loss_mean": metrics["loss/mean"],
            "loss_nan_count": metrics["loss/nan_count"],
            "gradient_norm": metrics["optim/grad_norm"],
            "time_per_step_seconds": metrics["time/step"],
            "renderer_enable_thinking": args.enable_thinking,
        },
        "output": {"model_path": str(output_model.resolve()), "model_sha256": output_sha},
    }
    receipt_path = state_dir / f"{args.run_name}-receipt.json"
    _write_once(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--optimizer-updates", type=int, default=1)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--uv-bin", type=Path, default=Path("/home/ubuntu/.local/bin/uv"))
    args = parser.parse_args()
    if not 0 < args.learning_rate <= 1e-4 or not 1 <= args.optimizer_updates <= 8:
        parser.error("learning rate or optimizer update count is outside the bounded range")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
