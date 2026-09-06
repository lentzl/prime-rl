from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from prime_rl.latent import h_iter_phase1_cap0 as contract

ROOT = Path(__file__).parents[3]
RUNNER_PATH = ROOT / "scripts/latent/run_h_iter_phase1_cap0_v1.py"
SPEC = importlib.util.spec_from_file_location("cap0_runner_test", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def load(path: Path) -> dict:
    data = path.read_bytes()
    value = contract.strict_loads(data[:-1])
    assert data == contract.canonical_json(value) + b"\n"
    return value


@pytest.fixture(scope="module")
def frozen() -> tuple[dict, dict, dict]:
    base = ROOT / "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-train-calibration-v1"
    selection = load(base / "cap0-probe-selection.json")
    capture = load(base / "capture-contract.json")
    value = load(ROOT / contract.ARTIFACT_DIR / "cap0-contract.json")
    contract.validate_contract(value, selection, capture)
    return value, selection, capture


def plan_fixture() -> dict:
    plan = {
        "schema_version": contract.PLAN_SCHEMA,
        "status": "preregistered",
        "mechanism": contract.MECHANISM,
        "run_identity": contract.RUN_ID,
        "mechanism_code_commit": "1" * 40,
        "execution_authorization": runner.AUTHORIZATION,
        "output_root": contract.OUTPUT_ROOT,
        "remote_paths": runner.REMOTE_PATHS,
        "runtime": contract.RUNTIME,
        "asset_sha256": {path: "2" * 64 for path in runner.PLAN_ASSET_PATHS},
        "mf0_archive_binding": contract.MF0_BINDING,
        "probe_contract": {"selection_sha256": contract.SELECTION_SHA256, "probe_count": 4, "repeat_count": 2, "model_forwards": 8, "tokenizer_calls": 4, "sequences": 192},
        "model_contract": {"e33_path": contract.E33_PATH, "e33_tree_sha256": contract.E33_TREE_SHA256, "e33_state_sha256": contract.E33_STATE_SHA256, "h176_path": contract.H176_PATH, "h176_tree_sha256": contract.H176_TREE_SHA256, "metadata_sha256": contract.METADATA_SHA256},
        "cache_contract": {"checks": 18, "mandatory_negative_trips": 1, "actual_allocations": 0, "pkv_none": True, "config_restored": True},
        "resource_bounds": contract.RESOURCE_BOUNDS,
        "memory_label_schedule": {"labels": contract.MEMORY_LABELS, "count": 28, "label_sha256": contract.sha256_bytes(contract.canonical_json(contract.MEMORY_LABELS))},
        "terminal_contract": runner.TERMINAL_CONTRACT,
        "safety_boundary": runner.SAFETY_BOUNDARY,
        "full_freeze": runner.FULL_FREEZE,
        "plan_sha256": "",
    }
    plan["plan_sha256"] = contract.sha256_bytes(contract.canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"}))
    runner.validate_plan(plan)
    return plan


def phase(phase_name: str, entered: int, exited: int, outcome: str) -> dict:
    caps = {"compute": 2700, "audit": 180, "failure_audit": 120, "terminal_publication": 60}
    cap = caps[phase_name] * 10**9
    return {"phase": phase_name, "entered_ns_since_start": entered, "exited_ns_since_start": exited, "duration_ns": exited - entered, "outcome": outcome, "cap_ns": cap, "alarm_after_ns": cap - 10**9, "alarm_safety_margin_ns": 10**9}


def terminal(entered: int) -> dict:
    return {"phase": "terminal_publication", "entered_ns_since_start": entered, "limit_ns": 60 * 10**9, "completion_observable_inside_terminal": False, "self_reference_boundary": "post_write_fsync_reopen_validation_and_process_exit_are_external_to_immutable_terminal_bytes"}


def memory_rows() -> list[dict]:
    return [{"label": label, "rss_bytes": index + 1, "peak_rss_bytes": index + 1, "allocated_bytes": 0, "reserved_bytes": 0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0} for index, label in enumerate(contract.MEMORY_LABELS)]


def cache_fixture(value: dict, complete: bool = True) -> dict:
    labels = contract.CACHE_LABELS if complete else ["CACHE_ENTRY"]
    return {"classes": value["cache_contract"]["class_closure"], "check_labels": labels, "check_count": len(labels), "trip_count": 1, "mandatory_negative_control": True, "actual_allocation_trips": 0, "pkv_non_none_count": 0, "config_drift_count": 0, "configuration_evidence": [], "classes_restored": complete, "configs_restored": complete, "complete": complete}


def probe_fixture(expected: dict) -> dict:
    repeats = []
    for repeat in (1, 2):
        repeats.append({"case_index": 2 * (expected["probe_index"] - 1) + repeat, "repeat": repeat, "model_call_index": 2 * (expected["probe_index"] - 1) + repeat, "input_ids_same_object": True, "attention_mask_same_object": True, "input_ids_sha256": "3" * 64, "attention_mask_sha256": "4" * 64, "logits_shape": [24, 1, 248320], "logits_dtype": "torch.bfloat16", "logits_finite": True, "logits_sha256": str(repeat) * 64, "full_hidden_shape": [24, 128, 2048], "full_hidden_dtype": "torch.bfloat16", "full_hidden_finite": True, "full_hidden_sha256": "5" * 64, "capture_shape": [24, 2048], "capture_dtype": "torch.bfloat16", "capture_finite": True, "capture_sha256": "6" * 64, "pkv_is_none": True})
    return {**{key: expected[key] for key in ("probe_index", "depth", "action_index", "replicate", "row_id", "row_sha256", "receiver_input_sha256")}, "node_count": 24, "local_text_byte_lengths": [68] * 24, "unpadded_token_lengths": [10] * 24, "input_ids_shape": [24, 128], "attention_mask_shape": [24, 128], "input_ids_sha256": "3" * 64, "attention_mask_sha256": "4" * 64, "repeats": repeats, "repeat_input_ids_bitwise": True, "repeat_attention_mask_bitwise": True, "repeat_full_hidden_bitwise": True, "repeat_capture_bitwise": True, "all_outputs_finite": True, "capture_row_sha256": [f"{index:064x}" for index in range(24)], "unique_capture_row_count": 24, "not_all_node_identical": True, "complete": True, "qualifies": True}


def proof_fixture(plan: dict, value: dict, selection: dict) -> dict:
    probes = [probe_fixture(row) for row in selection["ordered_probes"]]
    records = [phase("compute", 0, 1, "completed"), phase("audit", 1, 2, "completed")]
    proof = {
        "schema_version": contract.PROOF_SCHEMA, "status": contract.PROOF_STATUS, "mechanism": contract.MECHANISM, "run_identity": contract.RUN_ID,
        "execution_commit": "7" * 40, "mechanism_code_commit": plan["mechanism_code_commit"], "plan_file_sha256": "8" * 64, "plan_sha256": plan["plan_sha256"], "runtime": contract.RUNTIME,
        "asset_audit": {"before": plan["asset_sha256"], "after": plan["asset_sha256"], "equal": True}, "mf0_archive_binding": contract.MF0_BINDING,
        "selection": {"selection_sha256": contract.SELECTION_SHA256, "ordered_probes": selection["ordered_probes"]},
        "model_identity": {"class": contract.RUNTIME["model_class"], "hidden_size": 2048, "vocab_size": 248320, "dtype": "torch.bfloat16", "device": "cuda:0", "checkpoint": contract.E33_PATH, "checkpoint_tree_sha256": contract.E33_TREE_SHA256},
        "probes": probes, "aggregate": {"complete_modality_count": 4, "qualifying_modality_count": 4, "complete_probe_count": 4, "qualifying_probe_count": 4, "tokenizer_calls": 4, "model_forwards": 8, "sequences": 192, "all_qualify": True},
        "cache_guard": cache_fixture(value), "protected_state": {"disk_before": {"e33_tree_sha256": contract.E33_TREE_SHA256, "h176_tree_sha256": contract.H176_TREE_SHA256, "e33_metadata_sha256": contract.METADATA_SHA256, "h176_metadata_sha256": contract.METADATA_SHA256}, "disk_after": {"e33_tree_sha256": contract.E33_TREE_SHA256, "h176_tree_sha256": contract.H176_TREE_SHA256, "e33_metadata_sha256": contract.METADATA_SHA256, "h176_metadata_sha256": contract.METADATA_SHA256}, "e33_state_before": contract.E33_STATE_SHA256, "e33_state_after": contract.E33_STATE_SHA256, "e33_all_requires_grad_false": True, "e33_all_grads_none": True, "model_eval": True, "h176_loaded": False, "model_released": True},
        "safety": {"cuda_visible_devices": "0", "network_attempts": 0, "validation_opens": 0, "heldout_opens": 0, "generation_calls": 0, "backwards": 0, "optimizer_objects": 0, "optimizer_steps": 0, "candidate_objects": 0, "candidate_files": [], "checkpoint_files": [], "model_updated": False, "h176_loaded": False, "network_guard": {"installed": True, "wrappers_restored": False, "audit_hook_persistent": True, "attempt_count": 0, "operations": ["socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection", "socket.getaddrinfo"], "audit_events": ["socket.connect", "socket.getaddrinfo"]}, "experiment_open_firewall": {"denied_count": 0, "validation_open_count": 0, "heldout_open_count": 0, "opened_paths": []}},
        "counts": contract.COUNTS, "resources": {"bounds": contract.RESOURCE_BOUNDS, "gpu_name": contract.RUNTIME["gpu_model"], "gpu_total_bytes": 48 * 2**30, "host_ram_bytes": 64 * 2**30, "free_disk_bytes_preflight": 16 * 2**30, "free_disk_bytes_postflight": 16 * 2**30, "global_max_allocated_bytes": 0, "global_max_reserved_bytes": 0, "artifact_bytes_before_terminal": 0, "completed_phase_records": records, "final_terminal_publication": terminal(2), "prepublication_elapsed_ns": 2},
        "memory": {"labels": contract.MEMORY_LABELS, "label_sha256": contract.sha256_bytes(contract.canonical_json(contract.MEMORY_LABELS)), "rows": memory_rows()},
        "full_freeze": {"head_before": "7" * 40, "head_after": "7" * 40, "parent": plan["mechanism_code_commit"], "tree_before": "9" * 40, "tree_after": "9" * 40, "status_before": "", "status_after": "", "assets_equal": True},
        "tamper_audit": {"results": [{"name": name, "rejected": True, "error_type": "CAP0ContractError"} for name in contract.TAMPERS], "rejected_count": 42}, "decision_boundary": contract.DECISION, "proof_sha256": "",
    }
    proof["proof_sha256"] = contract.sha256_bytes(contract.canonical_json({key: item for key, item in proof.items() if key != "proof_sha256"}))
    return proof


def failure_fixture(plan: dict, value: dict, *, status: str, error_type: str, cause: str | None = None, exposure: bool = False) -> dict:
    records = [phase("compute", 0, 1, "error"), phase("failure_audit", 1, 2, "completed")]
    actual = {"validation_opens": 1 if exposure else 0, "heldout_opens": 0, "h176_loaded": False, "generation_calls": 0, "backwards": 0, "optimizer_objects": 0, "candidate_objects": 0, "network_attempts": 0, "e33_grads_present": False, "e33_state_changed": False, "positive_exposure": exposure, "network_guard": {"installed": True, "wrappers_restored": True, "audit_hook_persistent": True, "attempt_count": 0, "operations": ["socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection", "socket.getaddrinfo"], "audit_events": ["socket.connect", "socket.getaddrinfo"]}, "experiment_open_firewall": {"denied_count": 1 if exposure else 0, "validation_open_count": 1 if exposure else 0, "heldout_open_count": 0, "opened_paths": []}}
    progress = {"stage": "startup_pre_model", "current_probe": None, "current_repeat": None, "tokenizer_calls_completed": 0, "model_forwards_completed": 0, "sequences_completed": 0, "probes_completed": 0, "cache_checks_completed": 0, "memory_rows_completed": 0, "model_loaded": False, "model_released": False}
    cache = None
    if cause in {"cache_allocation_detected", "cache_configuration_drift", "returned_pkv_non_none"}:
        cache = cache_fixture(value, complete=False)
        progress["cache_checks_completed"] = 1
        progress.update({"stage": "capture", "model_loaded": True})
        if cause == "cache_allocation_detected":
            cache["trip_count"] = 2
            cache["actual_allocation_trips"] = 1
        elif cause == "cache_configuration_drift":
            cache["config_drift_count"] = 1
        else:
            cache["pkv_non_none_count"] = 1
            cache["check_labels"] = contract.CACHE_LABELS[:2]
            cache["check_count"] = 2
            progress.update({"stage": "capture", "current_probe": 1, "current_repeat": 1, "tokenizer_calls_completed": 1, "model_forwards_completed": 1, "sequences_completed": 24, "cache_checks_completed": 2, "model_loaded": True})
    audit = {"head": "7" * 40, "parent": plan["mechanism_code_commit"], "tree": "9" * 40, "status": "", "asset_hashes": plan["asset_sha256"], "execution_commit": "7" * 40, "errors": [], "exact": True}
    protected = {"disk_before": None, "disk_after": None, "e33_state_before": None, "e33_state_after": None, "grads_present": False, "restoration_attempted": True}
    if progress["model_loaded"]:
        disk = {"e33_tree_sha256": contract.E33_TREE_SHA256, "h176_tree_sha256": contract.H176_TREE_SHA256, "e33_metadata_sha256": contract.METADATA_SHA256, "h176_metadata_sha256": contract.METADATA_SHA256}
        protected.update({"disk_before": disk, "disk_after": disk, "e33_state_before": contract.E33_STATE_SHA256, "e33_state_after": contract.E33_STATE_SHA256})
    failure = {"schema_version": contract.FAILURE_SCHEMA, "status": status, "mechanism": contract.MECHANISM, "run_identity": contract.RUN_ID, "error_type": error_type, "error": "fixture", "traceback": "fixture", "execution_commit": "7" * 40, "mechanism_code_commit": plan["mechanism_code_commit"], "plan_file_sha256": "8" * 64, "plan_sha256": plan["plan_sha256"], "progress": progress, "partial_probes": [], "aggregate_partial": {"cause": cause, "probes_completed": 0, "nonfinite_observed": cause == "nonfinite_output"}, "cache_guard_partial": cache, "protected_state": protected, "actual_safety": actual, "resources": {"bounds": contract.RESOURCE_BOUNDS, "completed_phase_records": records, "final_terminal_publication": terminal(2), "prepublication_elapsed_ns": 2}, "partial_memory": {"labels": contract.MEMORY_LABELS, "rows": []}, "full_freeze_failure_audit": audit, "output_inventory_before_failure": [], "candidate_files": [], "checkpoint_files": [], "model_updated": False, "failure_sha256": ""}
    failure["failure_sha256"] = contract.sha256_bytes(contract.canonical_json({key: item for key, item in failure.items() if key != "failure_sha256"}))
    return failure


def test_contract_materializes_exact(frozen: tuple[dict, dict, dict]) -> None:
    value, selection, capture = frozen
    assert value == contract.build_contract(selection, capture)
    assert len(contract.TAMPERS) == 42 and len(contract.MEMORY_LABELS) == 28 and len(contract.CACHE_LABELS) == 18


def test_contract_tampers_1_through_40_rejected(frozen: tuple[dict, dict, dict]) -> None:
    value, selection, capture = frozen
    base = ROOT / "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-train-calibration-v1"
    bank = load(ROOT / "experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/train-bank.json")
    partition = load(base / "train-partition.json")
    results = runner.run_tampers(value, selection, capture, plan_fixture(), bank, partition)
    assert [row["name"] for row in results] == contract.TAMPERS[:40]


def test_plan_and_static_guard(frozen: tuple[dict, dict, dict]) -> None:
    plan = plan_fixture()
    assert list(plan["asset_sha256"]) == runner.PLAN_ASSET_PATHS
    reparsed = contract.strict_loads(contract.canonical_json(plan))
    runner.validate_plan(reparsed)
    assert reparsed["asset_sha256"] == plan["asset_sha256"]
    assert runner.static_guard(ROOT)["forbidden_sites"] == []
    assert not any("validation-bank" in path or "heldout-bank" in path for path in runner.PLAN_ASSET_PATHS)


def test_mf0_archive_and_train_only_inputs() -> None:
    assert runner.validate_archive(ROOT) == contract.MF0_BINDING
    bank, partition, selection, capture = runner.load_train_inputs(ROOT)
    assert len(bank["rows"]) == 96 and len(partition["fit_rows"]) == 64
    assert [row["row_id"] for row in selection["ordered_probes"]] == ["hi_e044b3b493fb81a4", "hi_d59ad4062a51dde8", "hi_06c7e038672bba2f", "hi_11171a28fbccbbd2"]
    contract.validate_contract(load(ROOT / contract.ARTIFACT_DIR / "cap0-contract.json"), selection, capture)


def test_strict_json_mapping_order_and_nonfinite() -> None:
    assert contract.canonical_json({"b": 2, "a": 1}) == contract.canonical_json({"a": 1, "b": 2})
    with pytest.raises(contract.CAP0ContractError): contract.strict_loads(b'{"a":1,"a":2}')
    with pytest.raises(contract.CAP0ContractError): contract.strict_loads(b'{"a":NaN}')


@pytest.mark.parametrize(("names", "maximum"), [(["compute", "audit"], 2880), (["compute", "failure_audit"], 2820), (["compute", "audit", "failure_audit"], 3000), (["compute", "audit", "terminal_publication", "failure_audit"], 3060)])
def test_timing_boundaries(names: list[str], maximum: int) -> None:
    outcomes = ["completed"] * len(names)
    if len(names) > 2:
        outcomes[-2] = "error"
    elif names == ["compute", "failure_audit"]:
        outcomes[0] = "error"
    rows = [phase(name, index, index + 1, outcomes[index]) for index, name in enumerate(names)]
    success = names == ["compute", "audit"]
    runner.validate_phase_records(rows, terminal(maximum * 10**9), maximum * 10**9, success)
    with pytest.raises(contract.CAP0ContractError): runner.validate_phase_records(rows, terminal(maximum * 10**9 + 1), maximum * 10**9 + 1, success)


def test_deep_proof_and_tampers(frozen: tuple[dict, dict, dict]) -> None:
    value, selection, _ = frozen
    plan = plan_fixture(); proof = proof_fixture(plan, value, selection)
    runner.validate_proof(proof, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)
    for key, changed_value in (("status", contract.REJECT_STATUS), ("counts", {}), ("proof_sha256", "0" * 64)):
        changed = copy.deepcopy(proof); changed[key] = changed_value
        if key != "proof_sha256": changed["proof_sha256"] = contract.sha256_bytes(contract.canonical_json({name: item for name, item in changed.items() if name != "proof_sha256"}))
        with pytest.raises(contract.CAP0ContractError): runner.validate_proof(changed, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)
    changed = copy.deepcopy(proof); changed["resources"]["global_max_allocated_bytes"] = 40 * 2**30 + 1
    changed["proof_sha256"] = contract.sha256_bytes(contract.canonical_json({name: item for name, item in changed.items() if name != "proof_sha256"}))
    with pytest.raises(contract.CAP0ContractError): runner.validate_proof(changed, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)


@pytest.mark.parametrize(("status", "error_type", "cause", "exposure"), [(contract.REJECT_STATUS, "CAP0MechanismRejected", "cache_allocation_detected", False), (contract.INCOMPLETE_STATUS, "CAP0ContractError", None, False), (contract.EXPOSURE_STATUS, "ExposureBoundaryRejected", None, True), (contract.INFRASTRUCTURE_STATUS, "InfrastructureInvalid", None, False)])
def test_every_failure_status(frozen: tuple[dict, dict, dict], status: str, error_type: str, cause: str | None, exposure: bool) -> None:
    value, selection, _ = frozen; plan = plan_fixture()
    failure = failure_fixture(plan, value, status=status, error_type=error_type, cause=cause, exposure=exposure)
    runner.validate_failure(failure, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)


@pytest.mark.parametrize("cause", ["cache_configuration_drift", "returned_pkv_non_none", "nonfinite_output", "repeat_parity_failed", "node_diversity_failed"])
def test_genuine_mechanism_rejection_evidence(frozen: tuple[dict, dict, dict], cause: str) -> None:
    value, selection, _ = frozen; plan = plan_fixture()
    failure = failure_fixture(plan, value, status=contract.REJECT_STATUS, error_type="CAP0MechanismRejected", cause=cause)
    if cause in {"repeat_parity_failed", "node_diversity_failed"}:
        failure["partial_probes"] = [probe_fixture(row) for row in selection["ordered_probes"]]
        failure["progress"].update({"stage": "postflight_audit", "tokenizer_calls_completed": 4, "model_forwards_completed": 8, "sequences_completed": 96, "probes_completed": 4, "cache_checks_completed": 18, "model_loaded": True})
        failure["aggregate_partial"]["probes_completed"] = 4
        failure["cache_guard_partial"] = cache_fixture(value)
        disk = {"e33_tree_sha256": contract.E33_TREE_SHA256, "h176_tree_sha256": contract.H176_TREE_SHA256, "e33_metadata_sha256": contract.METADATA_SHA256, "h176_metadata_sha256": contract.METADATA_SHA256}
        failure["protected_state"].update({"disk_before": disk, "disk_after": disk, "e33_state_before": contract.E33_STATE_SHA256, "e33_state_after": contract.E33_STATE_SHA256})
        if cause == "repeat_parity_failed":
            probe = failure["partial_probes"][0]
            probe["repeats"][1]["full_hidden_sha256"] = "a" * 64
            probe["repeat_full_hidden_bitwise"] = False
            probe["qualifies"] = False
        else:
            probe = failure["partial_probes"][0]
            probe["capture_row_sha256"] = ["a" * 64] * 24
            probe["unique_capture_row_count"] = 1
            probe["not_all_node_identical"] = False
            probe["qualifies"] = False
    failure["failure_sha256"] = contract.sha256_bytes(contract.canonical_json({key: item for key, item in failure.items() if key != "failure_sha256"}))
    runner.validate_failure(failure, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)


def test_exposure_precedence_and_progress_crossings(frozen: tuple[dict, dict, dict]) -> None:
    value, selection, _ = frozen; plan = plan_fixture()
    failure = failure_fixture(plan, value, status=contract.INCOMPLETE_STATUS, error_type="CAP0ContractError", exposure=True)
    with pytest.raises(contract.CAP0ContractError): runner.validate_failure(failure, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)


def test_failure_cache_exit_and_probe_underreport_rejected(frozen: tuple[dict, dict, dict]) -> None:
    value, selection, _ = frozen
    plan = plan_fixture()
    failure = failure_fixture(plan, value, status=contract.INCOMPLETE_STATUS, error_type="CAP0ContractError")
    failure["cache_guard_partial"] = cache_fixture(value)
    failure["progress"]["cache_checks_completed"] = 18
    failure["failure_sha256"] = contract.sha256_bytes(contract.canonical_json({key: item for key, item in failure.items() if key != "failure_sha256"}))
    with pytest.raises(contract.CAP0ContractError):
        runner.validate_failure(failure, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)


def test_returned_pkv_first_and_last_forward_prefixes(frozen: tuple[dict, dict, dict]) -> None:
    value, selection, _ = frozen
    plan = plan_fixture()
    first = failure_fixture(plan, value, status=contract.REJECT_STATUS, error_type="CAP0MechanismRejected", cause="returned_pkv_non_none")
    runner.validate_failure(first, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)

    last = failure_fixture(plan, value, status=contract.REJECT_STATUS, error_type="CAP0MechanismRejected", cause="returned_pkv_non_none")
    last["partial_probes"] = [probe_fixture(row) for row in selection["ordered_probes"][:3]]
    last["aggregate_partial"]["probes_completed"] = 3
    last["progress"].update({"current_probe": 4, "current_repeat": 2, "tokenizer_calls_completed": 4, "model_forwards_completed": 8, "sequences_completed": 96, "probes_completed": 3, "cache_checks_completed": 16})
    last["cache_guard_partial"]["check_labels"] = contract.CACHE_LABELS[:16]
    last["cache_guard_partial"]["check_count"] = 16
    last["failure_sha256"] = contract.sha256_bytes(contract.canonical_json({key: item for key, item in last.items() if key != "failure_sha256"}))
    runner.validate_failure(last, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)


def test_oom_and_protected_failure_closure(frozen: tuple[dict, dict, dict]) -> None:
    value, selection, _ = frozen
    plan = plan_fixture()
    oom = failure_fixture(plan, value, status=contract.INFRASTRUCTURE_STATUS, error_type="OutOfMemoryError")
    runner.validate_failure(oom, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)

    protected = failure_fixture(plan, value, status=contract.REJECT_STATUS, error_type="CAP0MechanismRejected", cause="returned_pkv_non_none")
    protected["protected_state"]["disk_after"] = None
    protected["failure_sha256"] = contract.sha256_bytes(contract.canonical_json({key: item for key, item in protected.items() if key != "failure_sha256"}))
    with pytest.raises(contract.CAP0ContractError):
        runner.validate_failure(protected, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)

    failure = failure_fixture(plan, value, status=contract.INCOMPLETE_STATUS, error_type="CAP0ContractError")
    failure["progress"].update({"stage": "capture", "tokenizer_calls_completed": 4, "model_forwards_completed": 8, "sequences_completed": 96, "cache_checks_completed": 17, "model_loaded": True})
    failure["cache_guard_partial"] = cache_fixture(value)
    failure["cache_guard_partial"]["check_labels"] = contract.CACHE_LABELS[:-1]
    failure["cache_guard_partial"]["check_count"] = 17
    failure["cache_guard_partial"].update({"classes_restored": False, "configs_restored": False, "complete": False})
    failure["failure_sha256"] = contract.sha256_bytes(contract.canonical_json({key: item for key, item in failure.items() if key != "failure_sha256"}))
    with pytest.raises(contract.CAP0ContractError):
        runner.validate_failure(failure, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)
    failure = failure_fixture(plan, value, status=contract.INCOMPLETE_STATUS, error_type="CAP0ContractError")
    failure["progress"]["sequences_completed"] = 24
    failure["failure_sha256"] = contract.sha256_bytes(contract.canonical_json({key: item for key, item in failure.items() if key != "failure_sha256"}))
    with pytest.raises(contract.CAP0ContractError): runner.validate_failure(failure, plan=plan, contract=value, selection=selection, execution_commit="7" * 40, plan_file_sha256="8" * 64)


def test_atomic_exclusive_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "cap0"
    monkeypatch.setattr(runner, "OUTPUT_ROOT", str(output))
    writer = runner.ArtifactWriter(output)
    assert writer.write("CAP0-PROOF.json", {"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
    with pytest.raises(runner.InfrastructureInvalid): writer.write("CAP0-FAILURE.json", {})


def test_launcher_modes_and_bounds() -> None:
    source = (ROOT / "scripts/latent/run_h_iter_phase1_cap0_v1.sh").read_text()
    assert 'timeout --signal=TERM --kill-after=30s 600s "$0" --inner "$@"' in source
    assert 'timeout --signal=TERM --kill-after=60s 3600s "$0" --inner "$@"' in source
    assert "--preflight-only" in source and "--validate-terminal" in source and contract.OUTPUT_ROOT in source
