from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prime_rl.latent import h_iter_phase1_mf0 as contract

ROOT = Path(__file__).parents[3]
RUNNER_PATH = ROOT / "scripts/latent/run_h_iter_phase1_mf0_v1.py"
SPEC = importlib.util.spec_from_file_location("mf0_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def load(path: Path) -> dict:
    data = path.read_bytes()
    value = json.loads(data)
    assert data == contract.canonical_json(value) + b"\n"
    return value


@pytest.fixture(scope="module")
def bank() -> dict:
    return load(ROOT / contract.TRAIN_BANK_PATH)


@pytest.fixture(scope="module")
def assets(bank: dict) -> dict[str, dict]:
    base = ROOT / contract.ARTIFACT_DIR
    observed = {name: load(base / name) for name in contract.ASSET_NAMES}
    contract.validate_assets(observed, bank)
    return observed


def plan_fixture() -> dict:
    plan = {
        "schema_version": runner.PLAN_SCHEMA,
        "status": "preregistered",
        "mechanism": contract.MECHANISM,
        "run_identity": contract.RUN_ID,
        "mechanism_code_commit": "1" * 40,
        "execution_authorization": {"mf0_model_free_prereg_only": True, "cap0": False, "t0": False, "model": False, "gpu": False, "training": False},
        "output_root": contract.OUTPUT_ROOT,
        "asset_sha256": {path: "2" * 64 for path in runner.PLAN_ASSET_PATHS},
        "runtime": runner.EXPECTED_RUNTIME,
        "resource_bounds": runner.RESOURCE_BOUNDS,
        "materialization_contract": {"asset_names": contract.ASSET_NAMES, "regenerate_byte_identical": True, "source_split": "train", "validation_or_heldout_paths_forbidden": True},
        "terminal_contract": {"success_file": "MF0-PROOF.json", "failure_file": "MF0-FAILURE.json", "exclusive_atomic": True, "canonical_roundtrip_twice": True, "success_status": runner.PROOF_STATUS, "failure_statuses": [runner.INCOMPLETE_STATUS, runner.EXPOSURE_STATUS, runner.INFRASTRUCTURE_STATUS]},
        "memory_label_schedule": {"labels": contract.MEMORY_LABELS, "label_sha256": contract.sha256_bytes(contract.canonical_json(contract.MEMORY_LABELS)), "count": 17},
        "safety_boundary": runner.SAFETY_BOUNDARY,
        "full_freeze": runner.FULL_FREEZE_CONTRACT,
        "plan_sha256": "",
    }
    plan["plan_sha256"] = contract.sha256_bytes(contract.canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"}))
    runner.validate_plan(plan)
    return plan


def phase_record(phase: str, entered: int, exited: int, outcome: str) -> dict:
    seconds = {"compute": 1050, "audit": 240, "failure_audit": 180, "terminal_publication": 60}[phase]
    cap = seconds * 1_000_000_000
    return {"phase": phase, "entered_ns_since_start": entered, "exited_ns_since_start": exited, "duration_ns": exited - entered, "outcome": outcome, "cap_ns": cap, "alarm_after_ns": cap - 1_000_000_000, "alarm_safety_margin_ns": 1_000_000_000, "timeout_observed": False, "alarm_requested_after_ns": cap - 1_000_000_000, "timeout_observed_duration_ns": None, "delivery_overrun_ns": 0, "timing_cap_exceeded": False}


def proof_fixture(plan: dict, assets: dict[str, dict]) -> dict:
    execution = "3" * 40
    rows = [{"label": label, "rss_bytes": index + 1, "peak_rss_bytes": index + 1} for index, label in enumerate(contract.MEMORY_LABELS)]
    synthetic_rows = [{"arm": arm, "output_shape": [4], "codec_gradient_nonzero": True, "codec_gradient_l2": 1.0, "readout_gradient_nonzero": True, "readout_gradient_l2": 1.0, "cell_gradient_nonzero": arm != "STATIC", "cell_gradient_l2": None if arm == "STATIC" else 1.0, "state_unchanged": True, "initial_tree_sha256": "4" * 64} for arm in contract.ARMS]
    synthetic = {"arms": synthetic_rows, "forwards": 5, "backwards": 5, "optimizer_objects": 0, "optimizer_steps": 0, "parameter_names": runner.EXPECTED_PARAMETER_NAMES, "parameter_count_per_arm": runner.EXPECTED_PARAMETER_COUNT, "initial_tree_sha256": "4" * 64, "all_initial_trees_equal": True, "synthetic_feature_shape": [24, 2048], "synthetic_feature_sha256": contract.SYNTHETIC_FEATURE_SHA256}
    census = {"transformers_modeling_modules": [], "pretrained_model_objects": 0, "tokenizer_objects": 0, "optimizer_objects": 0, "candidate_module_objects": 0, "uninspectable_count": 0, "census_errors": [], "cuda_initialized": False, "output_inventory": [], "object_census_method": "gc_mro_scan_without_importing_model_tokenizer_or_optimizer_classes"}
    network = {**runner.NETWORK_CONTRACT, "installed": True, "wrappers_restored": True, "audit_hook_persistent": True, "attempt_count": 0}
    safety = {"cuda_visible_devices": "", "cuda_initialized_before": False, "cuda_initialized_after": False, "torch_cpu_only": True, "tokenizer_calls": 0, "model_calls": 0, "model_backwards": 0, "optimizer_objects": 0, "optimizer_steps": 0, "validation_opens": 0, "heldout_opens": 0, "model_or_tokenizer_loaded": False, "candidate_created": False, "checkpoint_created": False, "model_updated": False, "object_inventory": census, "network_guard": network, "open_firewall": {"denied_count": 0, "validation_open_count": 0, "heldout_open_count": 0, "opened_paths": runner.EXPERIMENT_PLAN_ASSET_PATHS}, "static_guard": {"paths": ["src/prime_rl/latent/h_iter_phase1_mf0.py", "scripts/latent/run_h_iter_phase1_mf0_v1.py"], "forbidden_sites": [], "allowed_synthetic_backward_sites": ["site"]}}
    records = [phase_record("compute", 0, 10, "completed"), phase_record("audit", 10, 20, "completed")]
    terminal = {"phase": "terminal_publication", "entered_ns_since_start": 20, "limit_ns": 60_000_000_000, "completion_observable_inside_terminal": False, "self_reference_boundary": "post_write_fsync_reopen_validation_and_process_exit_are_external_to_immutable_terminal_bytes"}
    proof = {
        "schema_version": runner.PROOF_SCHEMA, "status": runner.PROOF_STATUS, "mechanism": contract.MECHANISM, "run_identity": contract.RUN_ID,
        "execution_commit": execution, "mechanism_code_commit": plan["mechanism_code_commit"], "plan_file_sha256": "5" * 64, "plan_sha256": plan["plan_sha256"], "runtime": runner.EXPECTED_RUNTIME,
        "asset_audit": {"before": plan["asset_sha256"], "after": plan["asset_sha256"], "regenerated": plan["asset_sha256"], "equal": True},
        "phase0_binding": assets["phase0-evidence-binding.json"], "train_partition": assets["train-partition.json"], "cap0_probe_selection": assets["cap0-probe-selection.json"], "training_schedule": assets["training-schedule.json"],
        "candidate_contract": {"contract": assets["candidate-module-contract.json"], "synthetic_validation": synthetic}, "capture_contract": assets["capture-contract.json"], "metric_gate_contract": assets["metric-gate-contract.json"], "threshold_builder_contract": assets["threshold-builder-contract.json"], "safety_resource_contract": plan["safety_boundary"],
        "tamper_audit": {"results": [{"name": name, "rejected": True, "error_type": "MF0ContractError"} for name in contract.TAMPERS], "rejected_count": 34}, "counts": runner.COUNTS, "safety": safety,
        "resources": {"bounds": runner.RESOURCE_BOUNDS, "host_ram_bytes": 8 * 2**30, "free_disk_bytes_preflight": 8 * 2**30, "free_disk_bytes_postflight": 8 * 2**30, "artifact_bytes_before_terminal": 0, "completed_phase_records": records, "final_terminal_publication": terminal, "prepublication_elapsed_ns": 20},
        "memory": {"labels": contract.MEMORY_LABELS, "label_sha256": contract.sha256_bytes(contract.canonical_json(contract.MEMORY_LABELS)), "rows": rows},
        "full_freeze": {"head_before": execution, "head_after": execution, "parent": plan["mechanism_code_commit"], "tree_before": "6" * 40, "tree_after": "6" * 40, "status_before": "", "status_after": "", "assets_equal": True}, "decision_boundary": runner.DECISION, "proof_sha256": "",
    }
    proof["proof_sha256"] = contract.sha256_bytes(contract.canonical_json({key: value for key, value in proof.items() if key != "proof_sha256"}))
    return proof


def failure_fixture(plan: dict) -> dict:
    execution = "3" * 40
    records = [phase_record("compute", 0, 1, "error"), phase_record("failure_audit", 1, 2, "completed")]
    terminal = {"phase": "terminal_publication", "entered_ns_since_start": 3, "limit_ns": 60_000_000_000, "completion_observable_inside_terminal": False, "self_reference_boundary": "post_write_fsync_reopen_validation_and_process_exit_are_external_to_immutable_terminal_bytes"}
    audit = {"head": execution, "head_exact": True, "parent": plan["mechanism_code_commit"], "parent_exact": True, "status": "", "status_clean": True, "plan_file_sha256": "5" * 64, "plan_file_exact": True, "plan_sha256": plan["plan_sha256"], "plan_internal_exact": True, "asset_hashes": plan["asset_sha256"], "assets_exact": True, "errors": [], "provenance_exact": True}
    failure = {"schema_version": runner.FAILURE_SCHEMA, "status": runner.INCOMPLETE_STATUS, "mechanism": contract.MECHANISM, "run_identity": contract.RUN_ID, "error_type": "MF0ContractError", "error": "fixture", "traceback": "fixture", "execution_commit": execution, "mechanism_code_commit": plan["mechanism_code_commit"], "plan_file_sha256": "5" * 64, "plan_sha256": plan["plan_sha256"], "completed_phase_records": records, "final_terminal_publication": terminal, "prepublication_elapsed_ns": 3, "progress": {"memory_rows": 0, "last_memory_label": None}, "actual_safety": {"cuda_visible_devices": "", "torch_imported": False, "object_inventory": None, "network_guard": {**runner.NETWORK_CONTRACT, "installed": False, "wrappers_restored": False, "audit_hook_persistent": False, "attempt_count": 0}, "open_firewall": None, "validation_opens": 0, "heldout_opens": 0}, "partial_memory": {"labels": contract.MEMORY_LABELS, "rows": []}, "full_freeze_failure_audit": audit, "output_inventory_before_failure": [], "candidate_created": False, "checkpoint_created": False, "model_or_tokenizer_loaded": False, "model_updated": False, "failure_sha256": ""}
    failure["failure_sha256"] = contract.sha256_bytes(contract.canonical_json({key: value for key, value in failure.items() if key != "failure_sha256"}))
    return failure


def failure_inventory() -> dict:
    return {"transformers_modeling_modules": [], "pretrained_model_objects": 0, "tokenizer_objects": 0, "optimizer_objects": 0, "candidate_module_objects": 0, "uninspectable_count": 0, "census_errors": [], "cuda_initialized": False, "output_inventory": [], "object_census_method": "gc_mro_scan_without_importing_model_tokenizer_or_optimizer_classes"}


def finish_failure(failure: dict) -> dict:
    failure["failure_sha256"] = contract.sha256_bytes(contract.canonical_json({key: value for key, value in failure.items() if key != "failure_sha256"}))
    return failure


def test_assets_regenerate_and_partition_exact(bank: dict, assets: dict[str, dict]) -> None:
    assert assets == contract.build_assets(bank)
    partition = assets["train-partition.json"]
    assert (len(partition["fit_rows"]), len(partition["calibration_rows"])) == (64, 32)
    assert {row["replicate"] for row in partition["fit_rows"]} == {0, 1, 2, 3}
    assert {row["replicate"] for row in partition["calibration_rows"]} == {4, 5}


def test_cap0_exact_minima(assets: dict[str, dict]) -> None:
    assert [row["row_id"] for row in assets["cap0-probe-selection.json"]["ordered_probes"]] == ["hi_e044b3b493fb81a4", "hi_d59ad4062a51dde8", "hi_06c7e038672bba2f", "hi_11171a28fbccbbd2"]


def test_sidecar_schedule_and_counts(assets: dict[str, dict]) -> None:
    schedule = assets["training-schedule.json"]
    operations = [*schedule["batches"]["preconnect"], *schedule["batches"]["precal"], *schedule["batches"]["train"], *schedule["batches"]["postcal"], *schedule["batches"]["postfit"]]
    assert [row["operation_index"] for row in operations] == list(range(385))
    assert schedule["expected_call_counts"]["sidecar_total"]["cell_calls"] == 773


def test_all_asset_tampers_rejected(bank: dict, assets: dict[str, dict]) -> None:
    for name in contract.TAMPERS[:-2]:
        with pytest.raises(contract.MF0ContractError):
            contract.validate_assets(runner.mutate_asset(assets, name), bank)


def test_strict_json_rejects_duplicates_and_nonfinite() -> None:
    with pytest.raises(contract.MF0ContractError): runner.strict_loads(b'{"x":1,"x":2}')
    with pytest.raises(contract.MF0ContractError): runner.strict_loads(b'{"x":NaN}')


def test_plan_forbids_validation_and_heldout_paths() -> None:
    plan = plan_fixture()
    plan["asset_sha256"]["validation-bank.json"] = "0" * 64
    plan["plan_sha256"] = contract.sha256_bytes(contract.canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"}))
    with pytest.raises(contract.MF0ContractError): runner.validate_plan(plan)


@pytest.mark.parametrize(
    ("phases", "outcomes", "success", "maximum"),
    [
        (["compute", "audit"], ["completed", "completed"], True, 1290),
        (["compute", "failure_audit"], ["error", "completed"], False, 1230),
        (["compute", "audit", "failure_audit"], ["completed", "error", "completed"], False, 1470),
        (["compute", "audit", "terminal_publication", "failure_audit"], ["completed", "completed", "error", "completed"], False, 1530),
    ],
)
def test_timing_exact_bound_and_plus_one_rejected(phases: list[str], outcomes: list[str], success: bool, maximum: int) -> None:
    records = [phase_record(phase, index, index + 1, outcomes[index]) for index, phase in enumerate(phases)]
    terminal = {"phase": "terminal_publication", "entered_ns_since_start": maximum * 1_000_000_000, "limit_ns": 60_000_000_000, "completion_observable_inside_terminal": False, "self_reference_boundary": "post_write_fsync_reopen_validation_and_process_exit_are_external_to_immutable_terminal_bytes"}
    runner.validate_phase_records(records, terminal, terminal["entered_ns_since_start"], success=success)
    terminal["entered_ns_since_start"] += 1
    with pytest.raises(contract.MF0ContractError): runner.validate_phase_records(records, terminal, terminal["entered_ns_since_start"], success=success)


def test_deep_proof_validator_and_tampers(assets: dict[str, dict]) -> None:
    plan = plan_fixture(); proof = proof_fixture(plan, assets)
    runner.validate_proof(proof, plan=plan, assets=assets, execution_commit="3" * 40, plan_file_sha256="5" * 64)
    for path, value in (("status", runner.INCOMPLETE_STATUS), ("counts", {}), ("proof_sha256", "0" * 64)):
        changed = copy.deepcopy(proof); changed[path] = value
        if path != "proof_sha256": changed["proof_sha256"] = contract.sha256_bytes(contract.canonical_json({key: item for key, item in changed.items() if key != "proof_sha256"}))
        with pytest.raises(contract.MF0ContractError): runner.validate_proof(changed, plan=plan, assets=assets, execution_commit="3" * 40, plan_file_sha256="5" * 64)
    changed = copy.deepcopy(proof); changed["candidate_contract"]["synthetic_validation"]["parameter_count_per_arm"] -= 1
    changed["proof_sha256"] = contract.sha256_bytes(contract.canonical_json({key: item for key, item in changed.items() if key != "proof_sha256"}))
    with pytest.raises(contract.MF0ContractError): runner.validate_proof(changed, plan=plan, assets=assets, execution_commit="3" * 40, plan_file_sha256="5" * 64)


def test_deep_failure_validator_and_taxonomy() -> None:
    plan = plan_fixture(); failure = failure_fixture(plan)
    runner.validate_failure(failure, plan=plan, execution_commit="3" * 40, plan_file_sha256="5" * 64)
    changed = copy.deepcopy(failure); changed["status"] = runner.INFRASTRUCTURE_STATUS
    changed["failure_sha256"] = contract.sha256_bytes(contract.canonical_json({key: value for key, value in changed.items() if key != "failure_sha256"}))
    with pytest.raises(contract.MF0ContractError): runner.validate_failure(changed, plan=plan, execution_commit="3" * 40, plan_file_sha256="5" * 64)


def test_failure_accepts_genuine_cuda_model_and_forbidden_open_exposure() -> None:
    plan = plan_fixture()
    for kind in ("cuda", "model", "forbidden_open"):
        failure = failure_fixture(plan)
        failure["status"] = runner.EXPOSURE_STATUS
        failure["error_type"] = "ExposureBoundaryRejected"
        failure["actual_safety"]["torch_imported"] = True
        failure["actual_safety"]["object_inventory"] = failure_inventory()
        if kind == "cuda":
            failure["actual_safety"]["object_inventory"]["cuda_initialized"] = True
        elif kind == "model":
            failure["actual_safety"]["object_inventory"]["pretrained_model_objects"] = 1
            failure["model_or_tokenizer_loaded"] = True
        else:
            failure["actual_safety"]["open_firewall"] = {"denied_count": 2, "validation_open_count": 1, "heldout_open_count": 1}
            failure["actual_safety"]["validation_opens"] = 1
            failure["actual_safety"]["heldout_opens"] = 1
        finish_failure(failure)
        runner.validate_failure(failure, plan=plan, execution_commit="3" * 40, plan_file_sha256="5" * 64)


def test_failure_exposure_cross_field_mismatches_rejected() -> None:
    plan = plan_fixture()
    no_evidence = failure_fixture(plan)
    no_evidence["status"] = runner.EXPOSURE_STATUS
    no_evidence["error_type"] = "ExposureBoundaryRejected"
    finish_failure(no_evidence)
    cuda_wrong_status = failure_fixture(plan)
    cuda_wrong_status["actual_safety"]["torch_imported"] = True
    cuda_wrong_status["actual_safety"]["object_inventory"] = failure_inventory()
    cuda_wrong_status["actual_safety"]["object_inventory"]["cuda_initialized"] = True
    finish_failure(cuda_wrong_status)
    model_mismatch = copy.deepcopy(cuda_wrong_status)
    model_mismatch["status"] = runner.EXPOSURE_STATUS
    model_mismatch["error_type"] = "ExposureBoundaryRejected"
    model_mismatch["actual_safety"]["object_inventory"]["cuda_initialized"] = False
    model_mismatch["actual_safety"]["object_inventory"]["tokenizer_objects"] = 1
    finish_failure(model_mismatch)
    count_mismatch = copy.deepcopy(model_mismatch)
    count_mismatch["actual_safety"]["object_inventory"]["tokenizer_objects"] = 0
    count_mismatch["actual_safety"]["open_firewall"] = {"denied_count": 1, "validation_open_count": 1, "heldout_open_count": 0}
    finish_failure(count_mismatch)
    firewall_mismatch = copy.deepcopy(count_mismatch)
    firewall_mismatch["actual_safety"]["validation_opens"] = 1
    firewall_mismatch["actual_safety"]["open_firewall"]["unexpected"] = 0
    finish_failure(firewall_mismatch)
    for failure in (no_evidence, cuda_wrong_status, model_mismatch, count_mismatch, firewall_mismatch):
        with pytest.raises(contract.MF0ContractError):
            runner.validate_failure(failure, plan=plan, execution_commit="3" * 40, plan_file_sha256="5" * 64)


def test_failure_census_errors_force_infrastructure_invalid() -> None:
    plan = plan_fixture()
    failure = failure_fixture(plan)
    failure["status"] = runner.INFRASTRUCTURE_STATUS
    failure["error_type"] = "InfrastructureInvalid"
    failure["actual_safety"]["torch_imported"] = True
    failure["actual_safety"]["object_inventory"] = failure_inventory()
    failure["actual_safety"]["object_inventory"]["uninspectable_count"] = 1
    failure["actual_safety"]["object_inventory"]["census_errors"] = [{"object_index": 1, "error_type": "RuntimeError", "error": "fixture"}]
    finish_failure(failure)
    runner.validate_failure(failure, plan=plan, execution_commit="3" * 40, plan_file_sha256="5" * 64)
    failure["status"] = runner.INCOMPLETE_STATUS
    failure["error_type"] = "MF0ContractError"
    finish_failure(failure)
    with pytest.raises(contract.MF0ContractError):
        runner.validate_failure(failure, plan=plan, execution_commit="3" * 40, plan_file_sha256="5" * 64)


def test_atomic_writer_exclusive_and_canonical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "run"; monkeypatch.setattr(runner, "OUTPUT_ROOT", str(output))
    writer = runner.ArtifactWriter(output); payload = {"b": 2, "a": 1}
    assert writer.write("MF0-PROOF.json", payload, 1024) == b'{"a":1,"b":2}\n'
    with pytest.raises(runner.InfrastructureInvalid): writer.write("MF0-FAILURE.json", payload, 1024)


def test_launcher_has_full_timeout_and_fresh_validator() -> None:
    source = (ROOT / "scripts/latent/run_h_iter_phase1_mf0_v1.sh").read_text()
    assert 'timeout --signal=TERM --kill-after=60s 1800s "$0" --inner "$@"' in source
    assert "--validate-terminal" in source
    assert contract.OUTPUT_ROOT in source


def test_static_guard_allows_only_the_synthetic_backward() -> None:
    evidence = runner.static_guard(ROOT)
    assert evidence["forbidden_sites"] == []
    assert len(evidence["allowed_synthetic_backward_sites"]) == 1


def test_runtime_mismatch_is_infrastructure_and_cuda_is_exposure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCuda:
        initialized = False

        @classmethod
        def is_initialized(cls) -> bool:
            return cls.initialized

    class FakeTorch:
        __version__ = "fixture"
        cuda = FakeCuda()

    monkeypatch.setattr(runner, "file_sha256", lambda _path: "fixture")
    observed = {"python": runner.platform.python_version(), "torch": "fixture", "sys_executable": sys.executable, "sys_prefix": sys.prefix, "shared_project_pyproject_sha256": "fixture", "shared_project_uv_lock_sha256": "fixture", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_initialized_required": False}
    monkeypatch.setattr(runner, "EXPECTED_RUNTIME", {**observed, "python": "different"})
    with pytest.raises(runner.InfrastructureInvalid): runner.validate_runtime(FakeTorch())
    monkeypatch.setattr(runner, "EXPECTED_RUNTIME", observed); FakeCuda.initialized = True
    with pytest.raises(runner.ExposureBoundaryRejected): runner.validate_runtime(FakeTorch())


def test_fresh_process_deep_validator(tmp_path: Path, assets: dict[str, dict]) -> None:
    plan = plan_fixture(); proof = proof_fixture(plan, assets)
    bundle = tmp_path / "bundle.json"
    bundle.write_bytes(contract.canonical_json({"plan": plan, "assets": assets, "proof": proof}))
    code = f'''\nimport importlib.util,json\nfrom pathlib import Path\np=Path({str(RUNNER_PATH)!r});s=importlib.util.spec_from_file_location("r",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nb=json.loads(Path({str(bundle)!r}).read_bytes())\nm.validate_proof(b["proof"],plan=b["plan"],assets=b["assets"],execution_commit="{'3'*40}",plan_file_sha256="{'5'*64}")\n'''
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)
