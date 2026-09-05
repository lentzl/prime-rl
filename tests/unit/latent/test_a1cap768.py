import ast
import copy
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

import prime_rl.latent.a1cap768 as cap
import prime_rl.latent.a1nc0 as a1nc0
from prime_rl.latent.a0 import canonical_json_hash


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_selection_schedule_memory_and_prior_evidence_are_exact():
    assert canonical_json_hash(cap.SELECTION) == cap.SELECTION_SHA256
    assert canonical_json_hash(cap.build_schedule()) == cap.SCHEDULE_SHA256
    assert len(cap.build_schedule()) == 32
    assert len(cap.memory_labels()) == len(set(cap.memory_labels())) == 67
    assert cap._validate_prior_evidence(cap.Path(".")) == cap.PRIOR_EVIDENCE
    assert cap._validate_launcher_rejection_evidence(cap.Path(".")) == cap.LAUNCHER_REJECTION_EVIDENCE
    assert cap._validate_import_rejection_evidence(cap.Path(".")) == cap.IMPORT_REJECTION_EVIDENCE
    assert cap._validate_import_proof_evidence(cap.Path(".")) == cap.IMPORT_PROOF_EVIDENCE
    assert (
        cap._validate_operational_render_rejection_evidence(cap.Path("."))
        == cap.OPERATIONAL_RENDER_REJECTION_EVIDENCE
    )
    assert cap.LAUNCHER_REJECTION_EVIDENCE["artifacts"] == []
    assert cap.LAUNCHER_REJECTION_EVIDENCE["shell_exit_nonzero"] is True
    assert "shell_exit_code" not in cap.LAUNCHER_REJECTION_EVIDENCE


def test_exact_host_import_proof_fails_closed_on_tamper(tmp_path):
    relative = Path("experiments/qwen35-2b-latent-workspace-v1")
    target = tmp_path / relative
    target.mkdir(parents=True)
    names = (
        "a1-nc0-cap768-r2-import-proof-receipt.json",
        "a1-nc0-cap768-r2-import-proof.log",
        "a1-nc0-cap768-r2-import-proof-exit-status.txt",
    )
    for name in names:
        shutil.copy2(relative / name, target / name)
    assert cap._validate_import_proof_evidence(tmp_path) == cap.IMPORT_PROOF_EVIDENCE
    (target / names[2]).write_text("1\n")
    with pytest.raises(ValueError, match="import-proof artifact changed"):
        cap._validate_import_proof_evidence(tmp_path)


def test_operational_render_rejection_fails_closed_on_tamper(tmp_path):
    relative = Path("experiments/qwen35-2b-latent-workspace-v1")
    target = tmp_path / relative
    target.mkdir(parents=True)
    names = (
        "a1-nc0-cap768-r2-operational-render-rejection-failure.json",
        "a1-nc0-cap768-r2-operational-render-rejection-run.log",
    )
    for name in names:
        shutil.copy2(relative / name, target / name)
    assert (
        cap._validate_operational_render_rejection_evidence(tmp_path)
        == cap.OPERATIONAL_RENDER_REJECTION_EVIDENCE
    )
    with (target / names[1]).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="operational-render rejection artifact changed"):
        cap._validate_operational_render_rejection_evidence(tmp_path)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (cap.CaptureMechanismRejected("x"), "capture768_mechanism_rejected"),
        (cap.ResourceFitRejected("x"), "capture768_resource_fit_rejected"),
        (cap.DiagnosticIncomplete("x"), "diagnostic_incomplete"),
        (ValueError("x"), "infrastructure_invalid"),
    ],
)
def test_terminal_taxonomy(error, status):
    assert cap.classify_failure(error)[0] == status


def test_cap_launcher_freezes_fresh_namespace_and_full_resource_bounds():
    shell = Path("scripts/latent/run_a1_nc0_cap768_v1.sh").read_text()
    assert "$3 != a1-nc0-cap768-run4" in shell
    assert "62914560" in shell and "67108864" in shell
    assert "CUDA_VISIBLE_DEVICES=0" in shell
    assert "3600s" in shell and "134217728" in shell


def test_launcher_declarations_execute_under_nounset_and_reject_reused_namespace():
    shell = Path("scripts/latent/run_a1_nc0_cap768_v1.sh").read_text()
    declarations = shell.split('\nfor asset in ', 1)[0]
    command = declarations + '\nprintf "%s\\n" "$shared_venv"'
    valid = subprocess.run(
        ["bash", "-c", command, "launcher", "a" * 40, "b" * 64, cap.AUTHORIZED_RUN_ID],
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0
    assert valid.stdout == "/home/ubuntu/rlm/prime-rl/.venv\n"
    assert "unbound variable" not in valid.stderr
    reused = subprocess.run(
        ["bash", "-c", declarations, "launcher", "a" * 40, "b" * 64, "a1-nc0-cap768-run3"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert reused.returncode == 64


def test_runner_imports_bank_validator_from_its_defining_module():
    tree = ast.parse(Path("scripts/latent/run_a1_nc0_cap768_v1.py").read_text())
    imports = {
        node.module: {alias.name for alias in node.names}
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "validate_bank_artifact" in imports["prime_rl.latent.a1nc0"]
    assert "validate_bank_artifact" not in imports["prime_rl.latent.a1cap768"]
    assert callable(a1nc0.validate_bank_artifact)


def test_r3_operational_render_repair_and_proof_are_narrow_and_strict():
    base = Path("scripts/latent/run_a1_nc0_nomination_v1.py").read_text()
    cap_runner = Path("scripts/latent/run_a1_nc0_cap768_v1.py").read_text()
    proof = Path("scripts/latent/prove_a1_nc0_cap768_r3_render_v1.py").read_text()
    proof_shell = Path("scripts/latent/prove_a1_nc0_cap768_r3_render_v1.sh").read_text()
    assert "def operational_template_input_ids(encoded: object) -> torch.Tensor:" in base
    for predicate in (
        "isinstance(encoded, BatchEncoding)",
        "ids.ndim != 2",
        "ids.shape[0] != 1",
        "ids.shape[1] <= 0",
        "ids.dtype != torch.long",
        'ids.device.type != "cpu"',
        "not ids.is_contiguous()",
    ):
        assert predicate in base
    assert "except base.ExperimentIncomplete as error:" in cap_runner
    assert 'raise DiagnosticIncomplete(f"CAP768 operational render failed: {error}")' in cap_runner
    assert "expected_lengths = [517, 475, 599, 471, 616, 476, 644, 470]" in proof
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != ""' in proof
    assert "tokenizer_load_calls != 1" in proof and "model_loader.call_count != 0" in proof
    assert "a1-nc0-cap768-r3-render-proof-run2" in proof_shell
    assert 'export CUDA_VISIBLE_DEVICES=""' in proof_shell
    assert "cap.validate_bank_artifact is not validate_bank_artifact" in proof
    assert 'validate_bank_artifact.__module__ != "prime_rl.latent.a1nc0"' in proof
    assert 'path = repo / "scripts/latent/run_a1_nc0_cap768_v1.py"' in proof
    assert "cap.base.operational_template_input_ids" in proof


def test_single_compute_alarm_covers_plan_through_probes():
    runner = Path("scripts/latent/run_a1_nc0_cap768_v1.py").read_text()
    main_start = runner.index("def main():")
    stage_start = runner.index('stage["compute_started"] = time.perf_counter()', main_start)
    plan_load = runner.index("plan = load_plan(args.plan, args.repo)", stage_start)
    run_call = runner.index("receipt = run(args, plan, writer, stage)", plan_load)
    assert stage_start < plan_load < run_call
    run_body = runner[runner.index("def run(") : main_start]
    assert 'started = float(stage["compute_started"])' in run_body
    assert 'signal.alarm(RESOURCE_BOUNDS["audit_seconds"])' in run_body
    assert "CAP768 preflight timeout" not in runner
    assert runner.count('signal.alarm(RESOURCE_BOUNDS["compute_seconds"])') == 1
    assert 'signal.alarm(RESOURCE_BOUNDS["failure_audit_seconds"])' in runner
    assert 'signal.alarm(RESOURCE_BOUNDS["terminal_seconds"])' in runner
    assert "signal.alarm(180)" not in runner and "signal.alarm(60)" not in runner


def valid_receipt():
    schedule = cap.build_schedule()
    calls = []
    probes = []
    for probe_index, selection in enumerate(cap.SELECTION, 1):
        modalities = {}
        for modality in ("PARENT", "MSELF"):
            unpadded = selection[f"{modality.lower()}_unpadded_tokens"]
            arms = [item["arm"] for item in schedule if item["probe_index"] == probe_index and item["modality"] == modality]
            operation_hashes = {
                arm: {
                    "last_logits_sha256": digest(f"{probe_index}:{modality}:logits"),
                    "full_hidden_sha256": digest(f"{probe_index}:{modality}:hidden"),
                    "capture_sha256": digest(f"{probe_index}:{modality}:capture"),
                }
                for arm in arms
            }
            case = {
                "unpadded_tokens": unpadded, "padded_tokens": 768, "padding_tokens": 768 - unpadded,
                "capture_indices": list(range(640, 768)), "capture_shape": [1, 128, 2048],
                "input_ids_sha256": digest("ids"), "attention_mask_sha256": digest("mask"),
                "captured_mask_sha256": digest("cmask"), "position_ids_sha256": digest("positions"),
                "exact_embeddings_sha256": digest("embeds"), "operation_hashes": operation_hashes,
                "full_hidden_sha256": next(iter(operation_hashes.values()))["full_hidden_sha256"],
                "capture_sha256": next(iter(operation_hashes.values()))["capture_sha256"],
                "keep1_logits_sha256": next(iter(operation_hashes.values()))["last_logits_sha256"],
                "embedding_lookup_cuda_event_seconds": 1.0, "embedding_lookup_wall_seconds": 1.0,
                "four_call_cuda_event_seconds": 4.0, "four_call_wall_seconds": 4.0,
                **{
                    key: True
                    for key in (
                        "exact_embeddings_finite", "exact_embeddings_requires_grad_false", "left_padding_exact",
                        "attention_mask_exact", "position_ids_exact", "no_truncation",
                        "id_embed_keep1_logits_bitwise", "id_embed_keep1_full_hidden_bitwise",
                        "id_embed_keep1_capture_bitwise", "repeat_same_embedding_object",
                        "repeat_embedding_unchanged", "repeat_logits_bitwise", "repeat_full_hidden_bitwise",
                        "repeat_capture_bitwise", "keep0_keep1_full_hidden_bitwise",
                        "keep0_keep1_capture_bitwise", "keep0_last_logits_keep1_bitwise", "all_outputs_finite",
                    )
                },
            }
            modalities[modality] = case
        probes.append({"selection": selection, "modalities": modalities})
    for item in schedule:
        selection = cap.SELECTION[item["probe_index"] - 1]
        calls.append(
            {
                **item,
                "unpadded_tokens": selection[f"{item['modality'].lower()}_unpadded_tokens"],
                "padded_tokens": 768,
                "logits_to_keep": 0 if item["arm"].endswith("KEEP0_CONTROL") else 1,
                "cuda_event_seconds": 1.0,
                "wall_seconds": 1.0,
                "logits_sha256": digest(f"{item['arm']}:logits"),
            }
        )
    labels = cap.memory_labels()
    rows = [
        {"label": label, "allocated_bytes": 0, "reserved_bytes": 0, "peak_allocated_bytes": 0,
         "peak_reserved_bytes": 0}
        for label in labels
    ]
    plan = {
        "plan_sha256": digest("plan"), "mechanism_code_commit": "1" * 40,
        "asset_sha256": {"scripts/latent/run_a1_nc0_cap768_v1.py": digest("runner")},
        "protected_checkpoints": {"coordinator_e33": cap._E33, "worker_h176": cap._H176},
        "prior_evidence": cap.PRIOR_EVIDENCE,
        "launcher_rejection_evidence": cap.LAUNCHER_REJECTION_EVIDENCE,
        "import_rejection_evidence": cap.IMPORT_REJECTION_EVIDENCE,
        "authorized_run_id": cap.AUTHORIZED_RUN_ID,
        "memory_labels": cap.memory_labels(),
    }
    physical = {"names": ["NVIDIA RTX A6000", "NVIDIA RTX A6000"], "uuids": ["GPU-0", "GPU-1"],
                "memory_used_mib": [100, 100], "compute_apps": []}
    preflight = {
        "enable_thinking": False, "tools_none_for_child": True, "parent_fixture_messages": 4,
        "child_base_messages": 2, "terminal_token_ids": [248046, 198],
        "fixed_continuation_token_ids": [49265, 48338, 3438, 321],
        "length_control_token_ids": [40, 4021, 2528, 8976, 35139, 635, 524, 599],
        "length_control_tokens_non_special": True, "tokenizer_eos_token_id": 248046,
        "tokenizer_pad_token_id": 248046, "maximum_unpadded_feature_tokens": 644,
        "feature_token_budget": 768, "feature_sequences_truncated": 0, "materialized_queries": 288,
        "tokenized_template_container": "transformers.tokenization_utils_base.BatchEncoding",
        "preflight_input_ids_extracted_from_batch_encoding": True,
        "batch_encoding_extraction_counts": {"parent": 96, "child_plain": 288, "child_opening": 288,
                                             "child_full": 288, "mself_parent": 288},
        "answer_key_interpolation_scope": "teacher_target_and_scoring_only",
        "answer_key_not_interpolated_into_parent_or_child_opening": True,
        "render_hashes_sha256": digest("renders"), "label_alignment_sha256": digest("alignment"),
    }
    per_probe = [
        {"probe_index": index, "embedding_cuda_event_seconds": 2.0, "embedding_wall_seconds": 2.0,
         "call_cuda_event_seconds": 8.0, "call_wall_seconds": 8.0}
        for index in range(1, 5)
    ]
    receipt = {
        "schema_version": cap.RECEIPT_SCHEMA, "status": "capture768_mechanism_validated",
        "plan_sha256": plan["plan_sha256"], "mechanism_code_commit": plan["mechanism_code_commit"],
        "execution_commit": "2" * 40, "asset_sha256": plan["asset_sha256"],
        "selection": cap.SELECTION, "selection_sha256": cap.SELECTION_SHA256,
        "call_schedule": schedule, "call_schedule_sha256": cap.SCHEDULE_SHA256,
        "prior_evidence": cap.PRIOR_EVIDENCE,
        "launcher_rejection_evidence": cap.LAUNCHER_REJECTION_EVIDENCE,
        "import_rejection_evidence": cap.IMPORT_REJECTION_EVIDENCE,
        "run_id": cap.AUTHORIZED_RUN_ID,
        "versions": {key: cap._RUNTIME[key] for key in ("python", "transformers", "flash_linear_attention",
                                                         "torch_distribution", "torch_runtime")},
        "runtime_sources": {
            name: {"path": f"/frozen/{name}.py", "sha256": sha}
            for name, sha in cap._RUNTIME["transformers_source_sha256"].items()
        },
        "static_guard": {"runner_sha256": digest("runner"), "forbidden_calls": []},
        "render_preflight": preflight,
        "protected_hashes_before": plan["protected_checkpoints"],
        "protected_hashes_after": plan["protected_checkpoints"],
        "checkpoint_metadata_before": {"coordinator_e33": cap._RUNTIME["checkpoint_metadata_sha256"],
                                       "worker_h176": cap._RUNTIME["checkpoint_metadata_sha256"]},
        "checkpoint_metadata_after": {"coordinator_e33": cap._RUNTIME["checkpoint_metadata_sha256"],
                                      "worker_h176": cap._RUNTIME["checkpoint_metadata_sha256"]},
        "e33_state_tree_before": digest("tree"), "e33_state_tree_after": digest("tree"),
        "e33_parameters_frozen_no_grad": True, "worker_h176_loaded": False,
        "model_runtime": {"class": cap._RUNTIME["model_class"], "hidden_size": 2048, "vocab_size": 248320,
                          "dtype": "torch.bfloat16", "device": "cuda:0"},
        "probes": probes, "calls": calls,
        "no_cache_contract": {"calls": 32, "use_cache_false": True, "pkv_input_none": True,
                              "pkv_output_none": True, "rope_reset_every_call": True, "embedding_lookups": 8,
                              "model_config_use_cache": False, "generation_config_use_cache": False},
        "cache_guard": {"classes": cap._CACHE_CLASS_CLOSURE, "negative_control_dynamic_cache_tripped": True,
                        "closure_check_count": 67, "restored_in_finally": True},
        "memory_ledger": rows, "memory_labels_sha256": canonical_json_hash(labels),
        "resources": {"gpu_name": "NVIDIA RTX A6000", "total_gpu_memory_bytes": 47 * 2**30,
                      "allocator_cap_bytes": 40 * 2**30, "peak_allocated_bytes": 0,
                      "peak_reserved_bytes": 0, "host_ram_bytes": 64 * 2**30,
                      "free_disk_bytes_before": 60 * 2**30, "visible_cuda_devices": 1,
                      "physical_gpu1_unused": True, "physical_gpu_audit_before": physical,
                      "physical_gpu_audit_after": physical, "network_used": False},
        "timings": {"tokenizer_seconds": 1.0, "model_load_seconds": 1.0, "compute_seconds": 1.0,
                    "audit_seconds": 1.0, "call_cuda_event_seconds_sum": 32.0,
                    "call_wall_seconds_sum": 32.0, "embedding_cuda_event_seconds_sum": 8.0,
                    "embedding_wall_seconds_sum": 8.0, "per_probe": per_probe, "total_seconds": 4.0},
        "claim": "capture768_geometry_and_resource_fit_only", "training_authorized": False,
        "bridge_created": False, "optimizer_created": False, "backward_used": False,
        "checkpoint_created": False, "candidate_created": False, "generation_used": False,
        "model_update_attempted": False, "semantic_heldout_output": False,
        "reusable_hidden_persisted": False, "interpretation_boundary": cap.INTERPRETATION,
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    return receipt, plan


def test_positive_receipt_and_tampering_fail_closed():
    receipt, plan = valid_receipt()
    cap.validate_receipt(receipt, plan=plan)
    for mutate in (
        lambda item: item["calls"][0].__setitem__("unpadded_tokens", 1),
        lambda item: item["probes"][0]["modalities"]["PARENT"].__setitem__("all_outputs_finite", False),
        lambda item: item["resources"].__setitem__("physical_gpu1_unused", False),
        lambda item: item["timings"].__setitem__("call_cuda_event_seconds_sum", 31.0),
        lambda item: item.__setitem__("semantic_answer", "leak"),
    ):
        broken = copy.deepcopy(receipt)
        mutate(broken)
        broken["receipt_sha256"] = canonical_json_hash(broken, omitted_fields=("receipt_sha256",))
        with pytest.raises(cap.DiagnosticIncomplete):
            cap.validate_receipt(broken, plan=plan)
