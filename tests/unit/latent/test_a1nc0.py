import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file

import prime_rl.latent.a1nc0 as a1nc0
from prime_rl.latent.a0 import canonical_json_hash
from prime_rl.latent.a1nc0 import (
    CacheAllocationDetected,
    ExperimentIncomplete,
    MechanismRejected,
    build_disjointness_report,
    build_memory_ledger_paths,
    classify_failure,
    fixed_feature_inputs,
    load_plan,
    module_state_tree_sha256,
    nomination_gate_passes,
    summarize_arm_results,
    tensor_bytes_sha256,
    validate_bank_artifact,
    validate_disjointness_report,
    validate_receipt,
    validate_schedule_artifact,
    validation_gate_passes,
)

EXPERIMENT = Path("experiments/qwen35-2b-latent-workspace-v1")


class _AnswerEncoding:
    input_ids = [7]


class _FrozenAnswerTokenizer:
    def __call__(self, _text, *, add_special_tokens):
        assert add_special_tokens is False
        return _AnswerEncoding()


def _bank_paths():
    return {split: EXPERIMENT / f"a1-nc0-{split}-bank-v1.json" for split in ("train", "validation", "held_out")}


def test_exact_banks_schedule_and_disjointness_are_closed():
    artifacts = {split: validate_bank_artifact(path, split) for split, path in _bank_paths().items()}
    schedule = json.loads((EXPERIMENT / "a1-nc0-schedule-v1.json").read_text())
    validate_schedule_artifact(schedule, artifacts=artifacts)
    report = json.loads((EXPERIMENT / "a1-nc0-disjointness-v1.json").read_text())
    validate_disjointness_report(report, artifacts=artifacts)
    assert report == build_disjointness_report(artifacts)
    assert report["all_pairwise_intersections_zero"] is True
    paths = build_memory_ledger_paths(schedule)
    assert {name: len(labels) for name, labels in paths.items()} == {
        "validation_stop": 6376,
        "full_evaluation": 10153,
    }
    assert (
        canonical_json_hash(paths["validation_stop"])
        == "863238ceb620b613acf6de23f138b70a6c4c18b53ee2c1c0f19f3532f60bd520"
    )
    assert (
        canonical_json_hash(paths["full_evaluation"])
        == "54452f9bfaf644647a581eab4f1a2191b0ad13d234cba42a4ba566a0d0e76eb9"
    )
    assert all(len(labels) == len(set(labels)) for labels in paths.values())


def test_schedule_or_disjointness_mutation_fails_closed():
    artifacts = {split: validate_bank_artifact(path, split) for split, path in _bank_paths().items()}
    schedule = json.loads((EXPERIMENT / "a1-nc0-schedule-v1.json").read_text())
    schedule["memory_ledger_paths"]["validation_stop"]["labels"][1] = "wrong"
    with pytest.raises(ValueError):
        validate_schedule_artifact(schedule, artifacts=artifacts)
    report = build_disjointness_report(artifacts)
    report["all_pairwise_intersections_zero"] = False
    with pytest.raises(ValueError):
        validate_disjointness_report(report, artifacts=artifacts)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (CacheAllocationDetected("cache"), "mechanism_rejected"),
        (MechanismRejected("numeric", reason="numeric"), "mechanism_rejected"),
        (ExperimentIncomplete("incomplete"), "mechanism_rejected"),
        (TimeoutError("timeout"), "infrastructure_invalid"),
        (RuntimeError("runtime"), "infrastructure_invalid"),
    ],
)
def test_terminal_failure_taxonomy_has_no_sixth_status(error, status):
    assert classify_failure(error)[0] == status


def test_tensor_byte_hash_handles_scalars_bfloat16_vectors_and_matrices():
    tensors = [
        torch.tensor(0.001, dtype=torch.float32),
        torch.tensor(0.001, dtype=torch.bfloat16),
        torch.arange(7, dtype=torch.float32),
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
    ]
    for tensor in tensors:
        expected = hashlib.sha256(
            tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        assert tensor_bytes_sha256(tensor) == expected


def test_module_state_tree_hash_includes_registered_buffers():
    module = torch.nn.Linear(2, 2)
    module.register_buffer("state", torch.tensor([1.0]))
    before = module_state_tree_sha256(module)
    module.state.add_(1.0)
    assert module_state_tree_sha256(module) != before


def test_fixed_feature_inputs_are_exactly_left_padded_without_truncation():
    ids = torch.tensor([[4, 5, 6]], dtype=torch.long)
    padded, mask = fixed_feature_inputs(ids, pad_token_id=99, budget=8)
    assert padded.tolist() == [[99, 99, 99, 99, 99, 4, 5, 6]]
    assert mask.tolist() == [[0, 0, 0, 0, 0, 1, 1, 1]]
    with pytest.raises(ExperimentIncomplete):
        fixed_feature_inputs(torch.arange(9).unsqueeze(0), pad_token_id=99, budget=8)


def _metric_rows():
    families = ("keyed_numeric", "relational_join", "config_structure", "ownership_graph")
    rows = []
    expected = []
    for index in range(48):
        family = families[index // 12]
        query_id = f"q{index:02d}"
        expected.append((query_id, family))
        arms = {}
        for arm in ("M0", "MOTH", "MSELF", "MCUR", "ZERO", "NOISE"):
            generated_text = "answer" if arm == "MCUR" else "wrong"
            arms[arm] = {
                "exact_match": arm == "MCUR",
                "generated_text": generated_text,
                "expected_answer_sha256": hashlib.sha256(b"answer").hexdigest(),
                "answer_token_nll": 1.0 if arm == "MCUR" else 2.0,
                "answer_token_count": 1,
                "generated_text_sha256": hashlib.sha256(generated_text.encode()).hexdigest(),
                "generated_token_ids": [1] * 12,
                "fixed_decode_steps": 12,
            }
        rows.append({"query_id": query_id, "family": family, "arms": arms})
    return rows, expected


def test_metric_summary_records_all_architecture_contrasts_and_contamination_floor():
    rows, expected = _metric_rows()
    summary = summarize_arm_results(rows, expected_queries=expected)
    assert set(summary["architecture_contrasts"]) == {"OPE", "OME", "CAG", "SSG", "DSC"}
    assert summary["architecture_contrasts"]["CAG"]["exact_utility_count"] == 48
    assert summary["recovery_contamination"] == {
        "allowed_floor_count": 12,
        "moth_observed_count": 0,
        "noise_observed_count": 0,
    }
    tampered = copy.deepcopy(rows)
    tampered[0]["arms"].pop("NOISE")
    with pytest.raises(ExperimentIncomplete):
        summarize_arm_results(tampered, expected_queries=expected)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("complete_tasks",), 47),
        (("arm_exact", "MCUR"), 15),
        (("m0_to_mcur_recoveries",), 8),
        (("m0_to_mcur_regressions",), 3),
        (("m0_to_mcur_recovery_families",), ["keyed_numeric", "relational_join"]),
        (("mcur_vs_moth", "exact_wins"), 9),
        (("mcur_vs_moth", "exact_losses"), 4),
        (("mcur_vs_moth", "exact_net"), 6),
        (("mcur_vs_moth", "win_families"), ["keyed_numeric", "relational_join"]),
        (("mcur_vs_moth", "mean_answer_token_nll_improvement"), 0.019),
        (("mcur_vs_moth", "paired_nll_wins"), 29),
        (("mcur_vs_mself", "exact_wins"), 6),
        (("mcur_vs_mself", "exact_losses"), 4),
        (("mcur_vs_mself", "exact_net"), 3),
        (("mcur_vs_mself", "mean_answer_token_nll_improvement"), 0.009),
        (("mcur_vs_mself", "paired_nll_wins"), 26),
        (("mcur_vs_zero", "exact_net"), 5),
        (("mcur_vs_zero", "mean_answer_token_nll_improvement"), 0.019),
        (("mcur_vs_noise", "exact_net"), 5),
        (("mcur_vs_noise", "mean_answer_token_nll_improvement"), 0.019),
        (("recovery_contamination", "moth_observed_count"), 13),
        (("recovery_contamination", "noise_observed_count"), 13),
        (("mcur_exact_by_family", "keyed_numeric"), 1),
    ],
)
def test_each_heldout_nomination_boundary_fails_independently(path, value):
    rows, expected = _metric_rows()
    summary = summarize_arm_results(rows, expected_queries=expected)
    assert nomination_gate_passes(summary)
    target = summary
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not nomination_gate_passes(summary)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("complete_tasks",), 47),
        (("arm_exact", "MCUR"), 11),
        (("mcur_vs_moth", "exact_net"), 3),
        (("m0_to_mcur_recoveries",), 5),
        (("m0_to_mcur_regressions",), 4),
        (("m0_to_mcur_recovery_families",), ["keyed_numeric"]),
        (("mcur_vs_moth", "mean_answer_token_nll_improvement"), 0.019),
        (("mcur_vs_moth", "paired_nll_wins"), 29),
    ],
)
def test_each_validation_proceed_boundary_fails_independently(path, value):
    rows, expected = _metric_rows()
    summary = summarize_arm_results(rows, expected_queries=expected)
    assert validation_gate_passes(summary)
    target = summary
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not validation_gate_passes(summary)


def _frozen_split_rows(schedule, split):
    artifact = validate_bank_artifact(_bank_paths()[split], split)
    query_lookup = {
        query["query_id"]: (record, query) for record in artifact["bank"]["records"] for query in record["queries"]
    }
    rows = []
    for item in schedule["arm_orders"][split]:
        query_id = item["query_id"]
        record, query = query_lookup[query_id]
        family = record["family"]
        arms = {}
        for arm in ("M0", "MOTH", "MSELF", "MCUR", "ZERO", "NOISE"):
            generated_text = query["answer"] if arm == "MCUR" else "wrong"
            arms[arm] = {
                "exact_match": arm == "MCUR",
                "generated_text": generated_text,
                "expected_answer_sha256": hashlib.sha256(query["answer"].encode()).hexdigest(),
                "answer_token_nll": 1.0 if arm == "MCUR" else 2.0,
                "answer_token_count": 1,
                "generated_text_sha256": hashlib.sha256(generated_text.encode()).hexdigest(),
                "generated_token_ids": [1] * 12,
                "fixed_decode_steps": 12,
            }
        rows.append({"query_id": query_id, "evidence_id": record["evidence_id"], "family": family, "arms": arms})
    return rows


def _positive_receipt(tmp_path):
    schedule = json.loads((EXPERIMENT / "a1-nc0-schedule-v1.json").read_text())
    artifacts = {split: validate_bank_artifact(path, split) for split, path in _bank_paths().items()}
    candidate_path = tmp_path / "bridge-candidate.safetensors"
    tensors = {
        name: torch.zeros(shape if shape else (), dtype=torch.float32)
        for name, shape in a1nc0._BRIDGE["candidate_tensor_shapes"].items()
    }
    save_file(tensors, candidate_path, metadata={"schema": "prime-rl/latent-a1-nc0-candidate/v1"})
    tree = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        entry = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": tensor_bytes_sha256(tensor),
        }
        tree.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
        tree.update(b"\n")
    validation_rows = _frozen_split_rows(schedule, "validation")
    held_rows = _frozen_split_rows(schedule, "held_out")

    def digest(label):
        return hashlib.sha256(label.encode()).hexdigest()

    def capture(label, *, keep0=False):
        keep1 = digest(f"{label}:keep1")
        return {
            "unpadded_tokens": 128,
            "padded_tokens": 256,
            "tokens_truncated": 0,
            "input_ids_sha256": digest(f"{label}:ids"),
            "attention_mask_sha256": digest(f"{label}:mask"),
            "captured_hidden_sha256": digest(f"{label}:hidden"),
            "full_final_hidden_sha256": digest(f"{label}:full"),
            "keep1_logits_sha256": keep1,
            "captured_mask_sha256": digest(f"{label}:captured-mask"),
            "captured_token_indices": list(range(128, 256)),
            "captured_visible_tokens": 128,
            "captured_zero_left_padding": True,
            "captured_suffix_matches_final_hidden_bitwise": True,
            "capture_spec_sha256": digest("capture-spec"),
            "gpu_seconds": 0.01,
            "keep0_control": None
            if not keep0
            else {
                "full_hidden_bitwise_equal": True,
                "selected_hidden_bitwise_equal": True,
                "keep0_full_hidden_sha256": digest(f"{label}:full"),
                "keep0_last_logits_sha256": keep1,
                "keep1_logits_sha256": keep1,
                "last_logits_bitwise_equal": True,
            },
        }

    def bridge_evidence(label, *, slot_sha=None):
        return {
            "encoder_workspace_float32_sha256": digest(f"{label}:workspace"),
            "receiver_precast_float32_sha256": digest(f"{label}:precast"),
            "receiver_final_bfloat16_sha256": slot_sha or digest(f"{label}:slots"),
            "receiver_gate_applied_exactly_once": True,
            "gpu_seconds": 0.01,
        }

    def workspace(label, *, slot_sha=None):
        return {
            "feature": capture(label),
            "bridge": bridge_evidence(label, slot_sha=slot_sha),
            "feature_bridge_cuda_event_seconds": 0.02,
            "feature_bridge_wall_seconds": 0.03,
        }

    def suffix(query_id=None, *, control=False):
        value = {
            "first_active_label_index": 10,
            "logit_suffix_start": 9,
            "logits_to_keep": 2,
            "active_label_count": 1,
            "active_causal_pairs_sha256": digest("pairs"),
            "active_causal_pairs_unchanged": True,
            "terminal_ids_excluded_from_teacher_input": True,
            "full_logits_control": None,
        }
        if query_id is not None:
            value["query_id"] = query_id
        if control:
            loss_sha = digest("loss")
            value["full_logits_control"] = {
                "last_k_logits_bitwise_equal": True,
                "loss_bitwise_equal": True,
                "full_logits_sha256": digest("full-logits"),
                "suffix_logits_sha256": digest("suffix-logits"),
                "full_loss_sha256": loss_sha,
                "suffix_loss_sha256": loss_sha,
            }
        return value

    label_alignment = {}
    for split, artifact in artifacts.items():
        for record in artifact["bank"]["records"]:
            for query in record["queries"]:
                label_alignment[f"{split}:{query['query_id']}"] = {
                    "active_label_positions": [10],
                    "active_logit_positions": [9],
                    "raw_answer_token_ids": [7],
                    "terminal_token_ids": [248046, 198],
                    "all_other_labels_masked": True,
                }

    gradient_groups = {name: 1.0 for name in a1nc0._GRADIENT_GROUPS}
    probes = []
    for index, selected in enumerate(a1nc0._A0NC_REPEAT_SELECTION):
        shared_capture = capture(f"probe:{index}", keep0=index == 0)
        repeated_capture = capture(f"probe:{index}")
        repeated_capture["captured_hidden_sha256"] = shared_capture["captured_hidden_sha256"]
        shared_bridge = bridge_evidence(f"probe:{index}")
        repeated_bridge = bridge_evidence(f"probe:{index}")
        steps = []
        for step, token in enumerate([49265, 48338, 3438, 321], start=1):
            hard = digest(f"probe:{index}:{step}:hard")
            soft = digest(f"probe:{index}:{step}:soft")
            steps.append(
                {
                    "step": step,
                    "continuation_token_id": token,
                    "l_id_l_e_bitwise_equal": True,
                    "soft_repeat_bitwise_equal": True,
                    "l_id_logits_sha256": hard,
                    "l_e_logits_sha256": hard,
                    "soft_logits_sha256": soft,
                    "soft_repeat_logits_sha256": soft,
                    "soft_same_tensor_object_for_repeat": True,
                    "soft_input_unchanged_after_forwards": True,
                    "l_id_input_ids_sha256": digest(f"probe:{index}:{step}:ids"),
                    "l_e_inputs_embeds_sha256": digest(f"probe:{index}:{step}:embeds"),
                    "shared_soft_inputs_embeds_sha256": digest(f"probe:{index}:{step}:soft-input"),
                    "attention_mask_sha256": digest(f"probe:{index}:{step}:mask"),
                    "position_ids_sha256": digest(f"probe:{index}:{step}:positions"),
                }
            )
        probes.append(
            {
                **selected,
                "capture_repeat_bitwise": True,
                "capture": [shared_capture, repeated_capture],
                "bridge_repeat_bitwise": True,
                "bridge": [shared_bridge, repeated_bridge],
                "soft_span_active": True,
                "soft_span_differs_from_hard": True,
                "outside_soft_span_exact": True,
                "steps": steps,
                "answer_token_count": 1,
                "suffix_objective": suffix(control=index == 0),
                "loss": 1.0,
            }
        )

    expected_ids = {
        query_id.rsplit("-q", 1)[0] for update in schedule["train_updates"][:16] for query_id in update["query_ids"]
    }
    feature_cache = []
    for evidence_id in sorted(expected_ids):
        item = capture(f"train:{evidence_id}")
        item.update(
            {
                "evidence_id": evidence_id,
                "host_hidden_sha256": digest(f"host:{evidence_id}:hidden"),
                "host_mask_sha256": digest(f"host:{evidence_id}:mask"),
                "detached": True,
                "device": "cpu",
            }
        )
        feature_cache.append(item)
    updates = []
    for update in schedule["train_updates"]:
        evidence_ids = {query_id.rsplit("-q", 1)[0] for query_id in update["query_ids"]}
        updates.append(
            {
                "epoch": update["epoch"],
                "update_index": update["update_index"],
                "query_ids_sha256": canonical_json_hash(update["query_ids"]),
                "query_exposures": 12,
                "mean_loss": 1.0,
                "suffix_objectives": [suffix(query_id) for query_id in update["query_ids"]],
                "base_model_gradients_absent_after_each_row": True,
                "preclip_gradient_l2": 1.0,
                "gradient_group_l2": gradient_groups,
                "receiver_gate_gradient_finite_nonzero": True,
                "within_update_evidence_workspace_sha256": {
                    evidence_id: digest(f"update:{update['update_index']}:{evidence_id}")
                    for evidence_id in evidence_ids
                },
                "bridge_parameter_sha256_after": digest(f"update:{update['update_index']}:bridge"),
            }
        )

    def enrich_evaluation(rows, split):
        artifact = artifacts[split]
        record_by_id = {record["evidence_id"]: record for record in artifact["bank"]["records"]}
        canonical = {
            evidence_id: workspace(f"{split}:{evidence_id}", slot_sha=digest(f"{split}:{evidence_id}:slots"))
            for evidence_id in record_by_id
        }
        setup = []
        noise_hashes = {}
        for evidence_id in record_by_id:
            donor = artifact["moth_donors"][evidence_id]
            target_sha = canonical[evidence_id]["bridge"]["receiver_final_bfloat16_sha256"]
            noise_sha = digest(f"{split}:{evidence_id}:noise")
            noise_hashes[evidence_id] = noise_sha
            payload = f"q35-2b-a1-nc0-split-information-v1:noise|2671655313|{split}|{evidence_id}"
            setup.append(
                {
                    "evidence_id": evidence_id,
                    "moth_donor_evidence_id": donor,
                    "mcur": canonical[evidence_id],
                    "moth_canonical_source_sha256": canonical[donor]["bridge"]["receiver_final_bfloat16_sha256"],
                    "noise": {
                        "payload": payload,
                        "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                        "seed": a1nc0.noise_seed(split, evidence_id),
                        "target_bfloat16_sha256": target_sha,
                        "raw_float32_sha256": digest(f"{payload}:raw"),
                        "scaled_float32_sha256": digest(f"{payload}:scaled"),
                        "final_bfloat16_sha256": noise_sha,
                        "target_slot_norms_float32": [1.0] * 8,
                        "precast_slot_norms_float32": [1.0] * 8,
                        "postcast_slot_norms_float32": [1.0] * 8,
                        "zero_target_rows": [],
                        "precast_target_norm_relative_tolerance": a1nc0._NOISE_NORM_RELATIVE_TOLERANCE,
                    },
                }
            )
        reuse = {
            evidence_id: {
                "MCUR": [canonical[evidence_id]["bridge"]["receiver_final_bfloat16_sha256"]] * 3,
                "MOTH": [canonical[artifact["moth_donors"][evidence_id]]["bridge"]["receiver_final_bfloat16_sha256"]]
                * 3,
                "NOISE": [noise_hashes[evidence_id]] * 3,
            }
            for evidence_id in record_by_id
        }
        for row, order in zip(rows, schedule["arm_orders"][split], strict=True):
            row["arm_order"] = order["arms"]
            costs = {}
            decodes = {}
            for arm in a1nc0._ARMS:
                feature_bridge = None
                if arm == "MCUR":
                    feature_bridge = canonical[row["evidence_id"]]
                elif arm == "MOTH":
                    feature_bridge = canonical[artifact["moth_donors"][row["evidence_id"]]]
                elif arm == "MSELF":
                    feature_bridge = workspace(f"{split}:{row['query_id']}:self")
                costs[arm] = {
                    "cuda_event_gpu_seconds": 1.0,
                    "wall_seconds": 1.0,
                    "feature_forwards": 1 if feature_bridge is not None else 0,
                    "bridge_forwards": 1 if feature_bridge is not None else 0,
                    "receiver_forwards": 12,
                    "receiver_cuda_event_seconds": 0.5,
                    "feature_bridge": feature_bridge,
                }
                nll = row["arms"][arm]["answer_token_nll"]
                decodes[arm] = [
                    {
                        "step": step,
                        "prefix_length": 20 + step - 1,
                        "argmax_token_id": 1,
                        "appended_token_id": 1,
                        "forced_after_eos": False,
                        "terminal_selected": False,
                        "gold_token_id": 7 if step == 1 else None,
                        "gold_token_nll": nll if step == 1 else None,
                        "prefix_sha256": digest(f"{split}:{row['query_id']}:{arm}:{step}:prefix"),
                        "attention_mask_sha256": digest(f"{split}:{row['query_id']}:{arm}:{step}:mask"),
                        "position_ids_sha256": digest(f"{split}:{row['query_id']}:{arm}:{step}:positions"),
                        "logits_sha256": digest(f"{split}:{row['query_id']}:{arm}:{step}:logits"),
                    }
                    for step in range(1, 13)
                ]
            row["costs"] = costs
            row["decode_evidence"] = decodes
        expected = [(row["query_id"], row["family"]) for row in rows]
        summary = summarize_arm_results(rows, expected_queries=expected)
        summary["mself_compute_match"] = {
            "mcur_feature_forwards": 48,
            "mself_feature_forwards": 48,
            "mcur_bridge_forwards": 48,
            "mself_bridge_forwards": 48,
            "mcur_feature_input_tokens": 48 * 256,
            "mself_feature_input_tokens": 48 * 256,
            "receiver_forwards_each": 48 * 12,
            "mcur_cuda_event_gpu_seconds": 48.0,
            "mself_cuda_event_gpu_seconds": 48.0,
            "relative_gpu_seconds_difference": 0.0,
        }
        summary["workspace_reuse"] = reuse
        summary["operation_counts"] = {
            "captures": 160,
            "bridges": 160,
            "receiver_forwards": 3456,
            "canonical_captures_and_bridges": 16,
            "mcur_query_captures_and_bridges": 48,
            "moth_query_captures_and_bridges": 48,
            "mself_query_captures_and_bridges": 48,
            "e33_call_histogram": {
                f"{split}_MCUR_CANONICAL_SETUP_FEATURE": 16,
                f"{split}_MCUR_FEATURE": 48,
                f"{split}_MOTH_FEATURE": 48,
                f"{split}_MSELF_FEATURE": 48,
                **{f"{split}_{arm}_DECODE": 576 for arm in a1nc0._ARMS},
            },
        }
        return {"rows": rows, "setup": setup, "summary": summary}

    validation = enrich_evaluation(validation_rows, "validation")
    held = enrich_evaluation(held_rows, "held_out")
    protected = {"coordinator_e33": a1nc0._E33, "worker_h176": a1nc0._H176}
    plan = {
        "plan_sha256": "1" * 64,
        "mechanism_code_commit": "2" * 40,
        "asset_sha256": {"scripts/latent/run_a1_nc0_nomination_v1.py": digest("runner")},
        "protected_checkpoints": protected,
        "a0nc_success_evidence": {"status": "nocache_receiver_mechanism_validated"},
        "bank_disjointness": a1nc0._DISJOINTNESS,
        "runtime": a1nc0._RUNTIME,
        "cache_guard_contract": a1nc0._CACHE_GUARD_CONTRACT,
        "resource_bounds": a1nc0._RESOURCE_BOUNDS,
    }
    labels = schedule["memory_ledger_paths"]["full_evaluation"]["labels"]
    receipt = {
        "schema_version": a1nc0.A1NC0_RECEIPT_SCHEMA,
        "status": "a1_nc0_nominated",
        "plan_sha256": plan["plan_sha256"],
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "execution_commit": "4" * 40,
        "execution_commit_is_exact_child_of_mechanism": True,
        "asset_sha256": plan["asset_sha256"],
        "a0nc_success_evidence": plan["a0nc_success_evidence"],
        "bank_disjointness": {
            "file_sha256": a1nc0._DISJOINTNESS["file_sha256"],
            "report_sha256": a1nc0._DISJOINTNESS["report_sha256"],
            "all_pairwise_intersections_zero": True,
        },
        "static_no_generation_guard": {
            "runner_sha256": digest("runner"),
            "forbidden_calls": [],
            "generate_used": False,
            "prepare_inputs_for_generation_used": False,
            "torch_manual_seed_call_count": 1,
            "torch_cuda_manual_seed_all_call_count": 1,
            "compose_receiver_inputs_gate_values": [1.0],
            "receiver_gate_applied_by_bridge_then_compose_gate_one": True,
        },
        "versions": {
            key: a1nc0._RUNTIME[key] for key in ("python", "transformers", "torch_distribution", "torch_runtime")
        },
        "runtime_sources": {
            name: {"path": f"/frozen/{name}.py", "sha256": sha}
            for name, sha in a1nc0._RUNTIME["transformers_source_sha256"].items()
        },
        "render_preflight": {
            "enable_thinking": False,
            "tools_none_for_child": True,
            "parent_fixture_messages": 4,
            "child_base_messages": 2,
            "terminal_token_ids": [248046, 198],
            "fixed_continuation_token_ids": [49265, 48338, 3438, 321],
            "length_control_token_ids": [40, 4021, 2528, 8976, 35139, 635, 524, 599],
            "length_control_tokens_non_special": True,
            "tokenizer_eos_token_id": 248046,
            "tokenizer_pad_token_id": 248046,
            "maximum_unpadded_feature_tokens": 200,
            "feature_sequences_truncated": 0,
            "materialized_queries": 288,
            "answer_key_interpolation_scope": "teacher_target_and_scoring_only",
            "answer_key_not_interpolated_into_parent_or_child_opening": True,
            "render_hashes_sha256": digest("renders"),
            "label_alignment": label_alignment,
            "label_alignment_sha256": canonical_json_hash(label_alignment),
        },
        "protected_hashes_before": protected,
        "protected_hashes_after": protected,
        "checkpoint_metadata_before": {
            "coordinator_e33": a1nc0._RUNTIME["checkpoint_metadata_sha256"],
            "worker_h176": a1nc0._RUNTIME["checkpoint_metadata_sha256"],
        },
        "checkpoint_metadata_after": {
            "coordinator_e33": a1nc0._RUNTIME["checkpoint_metadata_sha256"],
            "worker_h176": a1nc0._RUNTIME["checkpoint_metadata_sha256"],
        },
        "model_runtime": {
            "class": a1nc0._RUNTIME["model_class"],
            "hidden_size": 2048,
            "vocab_size": a1nc0._RUNTIME["vocab_size"],
            "dtype": "torch.bfloat16",
            "device": "cuda:0",
        },
        "interpretation_boundary": a1nc0._INTERPRETATION,
        "a1_admission": False,
        "live_harness_authorized": False,
        "a2_authorized": False,
        "model_promotion_authorized": False,
        "worker_h176_loaded": False,
        "live_trajectory_count": 0,
        "a_plus_b_combined": False,
        "resume_used": False,
        "candidate_valid": True,
        "candidate_valid_only_with_this_exact_terminal_receipt": True,
        "claim": "A1-NC0 nomination-only no-cache bridge learnability",
        "bound_a0nc_dependency_valid_for_B_only": True,
        "bridge": {
            "config": {
                "schema_version": a1nc0._BRIDGE["schema_version"],
                **{
                    key: a1nc0._BRIDGE[key]
                    for key in (
                        "source_width",
                        "workspace_width",
                        "receiver_width",
                        "slots",
                        "attention_heads",
                        "initial_receiver_gate",
                    )
                },
            },
            "trainable_parameter_count": 1321217,
            "initialization_seed": a1nc0._SEEDS["bridge_init"],
            "torch_manual_seed_calls": 1,
            "torch_cuda_manual_seed_all_calls": 1,
            "parameter_sha256_initial": digest("initial"),
            "optimizer_created": True,
            "optimizer_updates": 64,
            "optimizer_destroyed_before_evaluation": True,
            "optimizer_state_persisted": False,
            "base_model_checkpoint_created": False,
            "parameter_tree_hash_schema": "sorted_state_dict_name_dtype_shape_tensor_sha256_lines/v1",
            "parameter_sha256_final": tree.hexdigest(),
            "candidate": {
                "name": candidate_path.name,
                "bytes": candidate_path.stat().st_size,
                "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                "contains_bridge_and_receiver_gate_only": True,
                "valid_only_with_exact_terminal_receipt": True,
                "promotion_authorized": False,
            },
        },
        "train_feature_cache": feature_cache,
        "training_updates": updates,
        "training_invariants": {
            "feature_cache_after": {
                item["evidence_id"]: {
                    "hidden_sha256": item["host_hidden_sha256"],
                    "mask_sha256": item["host_mask_sha256"],
                    "device": "cpu",
                }
                for item in feature_cache
            },
            "all_64_host_features_unchanged_after_768_exposures": True,
            "epoch_positive_gradient_groups": {str(epoch): sorted(a1nc0._GRADIENT_GROUPS) for epoch in range(1, 5)},
        },
        "pretraining_a0nc_repeat": {
            "selection": schedule["a0nc_repeat_selection"],
            "selection_sha256": a1nc0._A0NC_REPEAT_SELECTION_SHA256,
            "fixed_continuation_text": " Acknowledged and continuing safely.",
            "fixed_continuation_token_ids": [49265, 48338, 3438, 321],
            "length_control_token_ids": [40, 4021, 2528, 8976, 35139, 635, 524, 599],
            "probes": probes,
            "gradient_group_l2": gradient_groups,
            "bridge_parameter_sha256_before": digest("preprobe"),
            "bridge_parameter_sha256_after": digest("preprobe"),
            "base_model_gradients_absent": True,
            "optimizer_step": False,
            "e33_forward_count": 78,
            "e33_call_histogram": {
                "A0NC_REPEAT_CAPTURE": 4,
                "A0NC_REPEAT_CAPTURE_REPEAT": 4,
                "A0NC_REPEAT_L_ID": 16,
                "A0NC_REPEAT_L_E": 16,
                "A0NC_REPEAT_S": 16,
                "A0NC_REPEAT_S_REPEAT": 16,
                "A0NC_REPEAT_GRADIENT": 4,
                "A0NC_REPEAT_CAPTURE_KEEP0_CONTROL": 1,
                "A0NC_REPEAT_GRADIENT_FULL_LOGITS_CONTROL": 1,
            },
            "bridge_forward_count": 8,
            "bridge_call_histogram": {"A0NC_REPEAT_BRIDGE": 4, "A0NC_REPEAT_BRIDGE_REPEAT": 4},
        },
        "validation": {**validation, "proceed_gate_passed": True},
        "held_out": {**held, "nomination_gate_passed": True},
        "no_cache_call_contract": {
            "total_e33_forwards": 8142,
            "expected_e33_forwards": 8142,
            "use_cache_false_every_call": True,
            "past_key_values_input_none_every_call": True,
            "past_key_values_output_none_every_call": True,
            "generate_used": False,
            "prepare_inputs_for_generation_used": False,
            "rope_deltas_reset_before_every_call": True,
            "model_config_use_cache": False,
            "generation_config_use_cache": False,
        },
        "cache_guard": {
            "classes": a1nc0._CACHE_CLASS_CLOSURE,
            "negative_control_dynamic_cache_tripped": True,
            "closure_check_count": labels.index("cache_guard_audit_complete") + 2,
            "restored_in_finally": True,
        },
        "memory_ledger": [
            {
                "label": label,
                "allocated_bytes": 0,
                "reserved_bytes": 0,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
            }
            for label in labels
        ],
        "memory_ledger_path": "full_evaluation",
        "memory_ledger_labels_sha256": canonical_json_hash(labels),
        "resources": {
            "wall_seconds": 100.0,
            "compute_seconds": 90.0,
            "audit_seconds_before_receipt_materialization": 5.0,
            "gpu_name": a1nc0._RESOURCE_BOUNDS["gpu_model"],
            "total_gpu_memory_bytes": 48 * 2**30,
            "allocator_cap_bytes": 40 * 2**30,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "host_ram_bytes": 128 * 2**30,
            "free_disk_bytes_before": 100 * 2**30,
            "visible_cuda_device_count": 1,
            "launcher_verified_two_a6000_idle_before_gpu0_exposure": True,
            "physical_gpu1_unused": True,
            "network_used": False,
        },
        "e33_parameter_tree_sha256_before": digest("e33"),
        "e33_parameter_tree_sha256_after": digest("e33"),
        "e33_tensor_tree_hash_schema": "sorted_state_dict_name_dtype_shape_tensor_sha256_lines/v1",
        "e33_parameters_require_grad_false": True,
        "e33_gradients_absent": True,
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    return receipt, plan, schedule, artifacts, candidate_path


def test_positive_receipt_validation_reaches_eof(tmp_path):
    receipt, plan, schedule, artifacts, candidate_path = _positive_receipt(tmp_path)
    assert (
        validate_receipt(
            receipt,
            plan=plan,
            schedule=schedule,
            artifacts=artifacts,
            tokenizer=_FrozenAnswerTokenizer(),
            candidate_path=candidate_path,
        )
        is None
    )


def test_validation_stop_receipt_uses_4526_calls_and_short_memory_branch(tmp_path):
    receipt, plan, schedule, artifacts, candidate_path = _positive_receipt(tmp_path)
    for row in receipt["validation"]["rows"]:
        for result in row["arms"].values():
            result["exact_match"] = False
            result["generated_text"] = "wrong"
            result["generated_text_sha256"] = hashlib.sha256(b"wrong").hexdigest()
            result["answer_token_nll"] = 1.0
        for steps in row["decode_evidence"].values():
            steps[0]["gold_token_nll"] = 1.0
    expected = [(row["query_id"], row["family"]) for row in receipt["validation"]["rows"]]
    prior_summary = receipt["validation"]["summary"]
    receipt["validation"]["summary"] = summarize_arm_results(receipt["validation"]["rows"], expected_queries=expected)
    for key in ("mself_compute_match", "workspace_reuse", "operation_counts"):
        receipt["validation"]["summary"][key] = prior_summary[key]
    receipt["validation"]["proceed_gate_passed"] = False
    receipt["held_out"] = None
    receipt["status"] = "valid_not_nominated_validation"
    receipt["no_cache_call_contract"]["total_e33_forwards"] = 4526
    receipt["no_cache_call_contract"]["expected_e33_forwards"] = 4526
    labels = schedule["memory_ledger_paths"]["validation_stop"]["labels"]
    receipt["memory_ledger"] = [
        {
            "label": label,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
        for label in labels
    ]
    receipt["memory_ledger_path"] = "validation_stop"
    receipt["memory_ledger_labels_sha256"] = canonical_json_hash(labels)
    receipt["cache_guard"]["closure_check_count"] = labels.index("cache_guard_audit_complete") + 2
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    assert (
        validate_receipt(
            receipt,
            plan=plan,
            schedule=schedule,
            artifacts=artifacts,
            tokenizer=_FrozenAnswerTokenizer(),
            candidate_path=candidate_path,
        )
        is None
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "self_hash",
        "candidate_bytes",
        "candidate_name",
        "candidate_shape",
        "candidate_content",
        "memory_branch",
        "call_count",
        "cache_guard",
        "preprobe",
        "training_update",
        "render",
        "runtime_source",
        "protected_tree",
        "evaluation_counts",
        "workspace_reuse",
        "bridge_unchanged",
        "memory_negative",
        "memory_nonmonotonic",
        "claim_boundary",
        "answer_binding",
        "nll_recompute",
    ],
)
def test_receipt_candidate_terminal_and_branch_tampering_fail_closed(tmp_path, tamper):
    receipt, plan, schedule, artifacts, candidate_path = _positive_receipt(tmp_path)
    if tamper == "self_hash":
        receipt["receipt_sha256"] = "0" * 64
    elif tamper == "candidate_bytes":
        candidate_path.write_bytes(candidate_path.read_bytes() + b"x")
    elif tamper == "candidate_name":
        receipt["bridge"]["candidate"]["name"] = "wrong.safetensors"
    elif tamper in {"candidate_shape", "candidate_content"}:
        tensors = {
            name: torch.zeros(shape if shape else (), dtype=torch.float32)
            for name, shape in a1nc0._BRIDGE["candidate_tensor_shapes"].items()
        }
        if tamper == "candidate_shape":
            tensors["decoder.receiver_gate"] = torch.zeros((1,), dtype=torch.float32)
        else:
            tensors["decoder.receiver_gate"] = torch.tensor(1.0, dtype=torch.float32)
        save_file(tensors, candidate_path, metadata={"schema": "prime-rl/latent-a1-nc0-candidate/v1"})
        receipt["bridge"]["candidate"]["bytes"] = candidate_path.stat().st_size
        receipt["bridge"]["candidate"]["sha256"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    elif tamper == "memory_branch":
        receipt["memory_ledger_path"] = "validation_stop"
    elif tamper == "call_count":
        receipt["no_cache_call_contract"]["total_e33_forwards"] = 8140
    elif tamper == "cache_guard":
        receipt["cache_guard"]["negative_control_dynamic_cache_tripped"] = False
    elif tamper == "preprobe":
        receipt["pretraining_a0nc_repeat"]["probes"][0]["soft_span_active"] = False
    elif tamper == "training_update":
        receipt["training_updates"][1]["query_ids_sha256"] = "0" * 64
    elif tamper == "render":
        receipt["render_preflight"]["feature_sequences_truncated"] = 1
    elif tamper == "runtime_source":
        next(iter(receipt["runtime_sources"].values()))["sha256"] = "0" * 64
    elif tamper == "protected_tree":
        receipt["e33_parameter_tree_sha256_after"] = "0" * 64
    elif tamper == "evaluation_counts":
        receipt["validation"]["summary"]["operation_counts"]["receiver_forwards"] = 3455
    elif tamper == "workspace_reuse":
        next(iter(receipt["validation"]["summary"]["workspace_reuse"].values()))["MCUR"][1] = "0" * 64
    elif tamper == "bridge_unchanged":
        receipt["bridge"]["parameter_sha256_initial"] = receipt["bridge"]["parameter_sha256_final"]
    elif tamper == "memory_negative":
        receipt["memory_ledger"][1]["allocated_bytes"] = -1
    elif tamper == "memory_nonmonotonic":
        receipt["memory_ledger"][0]["peak_allocated_bytes"] = 1
    elif tamper == "claim_boundary":
        receipt["worker_h176_loaded"] = True
    elif tamper == "answer_binding":
        receipt["validation"]["rows"][0]["arms"]["MCUR"]["exact_match"] = False
    elif tamper == "nll_recompute":
        receipt["validation"]["rows"][0]["arms"]["MCUR"]["answer_token_nll"] = 1.5
    if tamper not in {"self_hash", "candidate_bytes"}:
        receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    with pytest.raises(ExperimentIncomplete):
        validate_receipt(
            receipt,
            plan=plan,
            schedule=schedule,
            artifacts=artifacts,
            tokenizer=_FrozenAnswerTokenizer(),
            candidate_path=candidate_path,
        )


def test_candidate_without_terminal_is_invalid_and_failure_can_follow(tmp_path):
    runner_path = Path("scripts/latent/run_a1_nc0_nomination_v1.py").resolve()
    spec = importlib.util.spec_from_file_location("a1nc0_runner_for_test", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    runner.OUTPUT_ROOT = output_root
    writer = runner.ArtifactWriter(output_root / "a1-nc0-test")
    try:
        candidate = writer.write_candidate({"decoder.receiver_gate": torch.tensor(0.0)}, 1024 * 1024)
        assert candidate["name"] == "bridge-candidate.safetensors"
        assert writer.terminal_name is None
        writer.write_terminal(
            "failure.json",
            {"status": "mechanism_rejected", "candidate_valid": False},
            1024 * 1024,
        )
        assert [item["name"] for item in writer.inventory()] == [
            "bridge-candidate.safetensors",
            "failure.json",
        ]
    finally:
        writer.close()


def test_launcher_idle_guard_does_not_match_its_own_script_name():
    shell = Path("scripts/latent/run_a1_nc0_nomination_v1.sh").read_text()
    guard = next(line for line in shell.splitlines() if line.startswith("! pgrep -af"))
    assert "run_a1_nc0" not in guard
    assert "[v]llm" in guard
    assert "[p]rime_rl" in guard


def test_frozen_plan_loads_with_exact_assets():
    plan = EXPERIMENT / "a1-nc0-plan-v1.json"
    if not plan.exists():
        pytest.skip("freeze commit not materialized yet")
    loaded, artifacts, schedule, disjointness = load_plan(
        plan,
        _bank_paths(),
        EXPERIMENT / "a1-nc0-schedule-v1.json",
        EXPERIMENT / "a1-nc0-disjointness-v1.json",
    )
    assert loaded["status"] == "preregistered"
    assert len(artifacts["train"]["bank"]["records"]) == 64
    assert schedule["train_schedule_sha256"].startswith("5c9c")
    assert disjointness["all_pairwise_intersections_zero"] is True
