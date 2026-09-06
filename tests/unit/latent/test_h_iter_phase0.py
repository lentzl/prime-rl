from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

import prime_rl.latent.h_iter_phase0 as hiter

ROOT = Path(__file__).parents[3]
ASSET_DIR = ROOT / hiter.ARTIFACT_DIR_REL


def load_asset(name: str) -> dict:
    data = (ASSET_DIR / name).read_bytes()
    assert data.endswith(b"\n") and not data.endswith(b"\n\n")
    value = hiter.strict_json_loads(data[:-1])
    assert data == hiter.canonical_json(value) + b"\n"
    return value


def test_generator_payloads_and_banks_regenerate_exactly() -> None:
    banks = {split: load_asset(f"{split}-bank.json") for split in hiter.SPLITS}
    summary = hiter.validate_banks(banks)
    assert {key: summary[key] for key in summary if key != "split_summaries"} == {
        "total_rows": 192,
        "unique_row_ids": 192,
        "unique_node_ids": 4608,
        "unique_nonces": 4608,
        "unique_receiver_input_sha256": 192,
    }
    assert [(row["split"], row["row_count"], row["structural_rows_valid"]) for row in summary["split_summaries"]] == [
        ("train", 96, 96),
        ("validation", 48, 48),
        ("heldout", 48, 48),
    ]
    assert all(len(row[key]) == 64 for row in summary["split_summaries"] for key in (
        "row_order_sha256", "node_serialization_order_sha256", "edge_serialization_order_sha256"
    ))
    for split in hiter.SPLITS:
        assert hiter.generate_bank(split) == banks[split]
        assert hiter._digest(hiter.BANK_PAYLOADS[split]) == hiter.EXPECTED_PAYLOAD_SHA256[split]
        assert hiter.seed_u64(hiter.BANK_PAYLOADS[split]) == hiter.EXPECTED_PAYLOAD_SEEDS[split]
        assert hiter._digest(hiter.ORDER_PAYLOADS[split]) == hiter.EXPECTED_ORDER_SHA256[split]
        assert hiter.seed_u64(hiter.ORDER_PAYLOADS[split]) == hiter.EXPECTED_ORDER_SEEDS[split]


def test_exact_generator_structure_and_answer_free_boundary() -> None:
    for split in hiter.SPLITS:
        bank = load_asset(f"{split}-bank.json")
        for row in bank["rows"]:
            ring, successor, local = hiter.row_ring(row)
            assert len(ring) == len(set(ring)) == 24
            assert successor[ring[-1]] == ring[0]
            assert row["supervision"]["target_node_id"] == ring[row["depth"]]
            markers = [hiter.marker_from_local_text(local[node_id]) for node_id in ring]
            assert [markers.count(marker) for marker in hiter.MARKERS] == [6, 6, 6, 6]
            donor, distance, marker = hiter.donor_for_row(row)
            assert donor in ring and distance > row["depth"]
            assert marker != row["supervision"]["target_marker"]
            encoded = hiter.canonical_json(row["receiver_input"]).decode()
            assert all(action not in encoded for action in hiter.ACTIONS)


def test_probe_and_operation_schedule_are_exact() -> None:
    banks = {split: load_asset(f"{split}-bank.json") for split in hiter.SPLITS}
    selection = load_asset("locality-probe-selection.json")
    schedule = load_asset("operation-schedule.json")
    hiter.validate_probe_selection(selection, banks)
    hiter.validate_schedule(schedule, selection)
    assert [probe["depth"] for probe in selection["probes"]] == list(range(1, 9))
    assert schedule["expected_counts"] == {
        "arm_executions": 128,
        "transition_calls": 348,
        "readout_calls": 144,
        "graph_encode_passes": 16,
        "local_node_encode_calls": 384,
        "synthetic_backward_calls": 40,
        "perturb_runs": 80,
        "endpoint_swap_runs": 8,
        "model_or_transformer_forwards": 0,
        "optimizer_steps": 0,
    }


def test_overlap_sources_reparse_to_recorded_empty_intersections() -> None:
    pytest.importorskip("pyarrow")
    banks = {split: load_asset(f"{split}-bank.json") for split in hiter.SPLITS}
    identities = hiter.new_identity_sets(banks)
    overlap = load_asset("overlap-evidence.json")
    assert overlap["source_record_count"] == 38
    assert overlap["all_intersections_empty"] is True
    for record in overlap["source_records"]:
        data = subprocess.run(
            ["git", "show", f"{record['source_commit']}:{record['source_path']}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        assert hiter.sha256_bytes(data) == record["file_sha256"]
        observed, intersection = hiter.extract_prior_source(record["source_path"], data, identities)
        assert observed == record["observed"]
        assert intersection == record["intersection"]
        assert not any(intersection.values())


def test_locality_numeric_and_symbolic_proofs(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    banks = {split: load_asset(f"{split}-bank.json") for split in hiter.SPLITS}
    selection = load_asset("locality-probe-selection.json")
    original_transition = hiter._transition
    actual_transition_calls = 0

    def counted_transition(hidden, successor_index):
        nonlocal actual_transition_calls
        actual_transition_calls += 1
        return original_transition(hidden, successor_index)

    monkeypatch.setattr(hiter, "_transition", counted_transition)
    locality = hiter.run_all_locality_probes(banks, selection)
    locality["policy"] = hiter.locality_policy()
    locality["symbolic_dependencies"] = hiter.run_symbolic_dependency_audit(banks)
    hiter.validate_locality_evidence(locality, selection)
    assert locality["counts"]["graph_encode_passes"] == 16
    assert locality["counts"]["local_node_encode_calls"] == 384
    assert actual_transition_calls == 348


def test_all_mechanism_tampers_are_rejected() -> None:
    pytest.importorskip("torch")
    banks = {split: load_asset(f"{split}-bank.json") for split in hiter.SPLITS}
    selection = load_asset("locality-probe-selection.json")
    locality = hiter.run_all_locality_probes(banks, selection)
    locality["policy"] = hiter.locality_policy()
    locality["symbolic_dependencies"] = hiter.run_symbolic_dependency_audit(banks)
    result = hiter.run_mechanism_tamper_audit(banks, selection, locality)
    assert result["rejected_count"] == 26
    assert [row["name"] for row in result["results"]] == hiter.MECHANISM_TAMPERS
    assert all(row["rejected"] for row in result["results"])


def test_strict_canonical_json_rejects_duplicates_and_nonfinite() -> None:
    with pytest.raises(hiter.ContractError, match="duplicate"):
        hiter.strict_json_loads('{"x":1,"x":2}')
    with pytest.raises(hiter.ContractError, match="nonfinite"):
        hiter.strict_json_loads('{"x":NaN}')
    with pytest.raises(ValueError):
        hiter.canonical_json({"x": float("inf")})


def test_transition_and_readout_source_expose_only_local_one_hop_state() -> None:
    source = Path(hiter.__file__).read_text()
    tree = ast.parse(source)
    transition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_transition")
    readout = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_readout")
    transition_text = ast.unparse(transition)
    readout_text = ast.unparse(readout)
    assert "index_select" in transition_text
    assert "matmul" not in transition_text and "adjacency" not in transition_text and "pool" not in transition_text
    assert [argument.arg for argument in readout.args.args] == ["vector"]
    assert "start_index" not in readout_text and "state" not in readout_text
    torch = pytest.importorskip("torch")
    with pytest.raises(hiter.ContractError, match="indexed start vector"):
        hiter._readout(torch.zeros((24, 4), dtype=torch.float64))


def test_tamper_schedule_and_memory_labels_are_stable() -> None:
    tamper = load_asset("tamper-schedule.json")
    assert tamper == hiter.build_tamper_schedule()
    assert [row["name"] for row in tamper["mechanism_tampers"]] == hiter.MECHANISM_TAMPERS
    assert [row["name"] for row in tamper["receipt_tampers"]] == hiter.RECEIPT_TAMPERS
    labels = hiter.memory_labels()
    assert len(labels) == len(set(labels)) == 26


def test_failure_schema_rejects_unsafe_and_stale_terminals() -> None:
    plan = load_asset("phase0-plan.json")
    plan["runtime"] = hiter.EXPECTED_RUNTIME
    plan["resource_bounds"] = hiter.RESOURCE_BOUNDS
    plan["safety_boundary"] = hiter.DECISION_BOUNDARY | {
        "coordinator_e33_loaded": False,
        "worker_h176_loaded": False,
        "tokenizer_calls": 0,
        "model_forwards": 0,
        "model_backwards": 0,
        "optimizer_steps": 0,
        "synthetic_cpu_backwards": 40,
        "transformers_modeling_imports": 0,
        "network_guard": hiter.NETWORK_GUARD_CONTRACT,
    }
    plan["plan_sha256"] = hiter.canonical_sha256(plan, omit="plan_sha256")
    execution = "1" * 40
    external_plan = "2" * 64
    def phase_record(phase: str, entered: int, exited: int, outcome: str) -> dict:
        cap = hiter.PHASE_CAP_SECONDS[phase] * 1_000_000_000
        return {
            "phase": phase,
            "entered_ns_since_start": entered,
            "exited_ns_since_start": exited,
            "duration_ns": exited - entered,
            "outcome": outcome,
            "cap_ns": cap,
            "alarm_after_ns": cap - 1_000_000_000,
            "alarm_safety_margin_ns": 1_000_000_000,
            "timeout_observed": False,
            "alarm_requested_after_ns": cap - 1_000_000_000,
            "timeout_observed_duration_ns": None,
            "delivery_overrun_ns": 0,
            "timing_cap_exceeded": False,
        }

    phase_records = [
        phase_record("compute", 0, 10, "error"),
        phase_record("failure_audit", 10, 20, "completed"),
    ]
    failure = {
        "schema_version": hiter.FAILURE_SCHEMA,
        "status": "h_iter_phase0_generator_locality_incomplete",
        "mechanism": hiter.MECHANISM,
        "run_identity": hiter.RUN_IDENTITY,
        "error_type": "ContractError",
        "error": "synthetic",
        "traceback": "synthetic traceback",
        "execution_commit": execution,
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "authorized_plan_file_sha256": external_plan,
        "plan_file_sha256": external_plan,
        "plan_sha256": plan["plan_sha256"],
        "actual_safety": {
            "cuda_visible_devices": "",
            "torch_imported": True,
            "cuda_initialized": False,
            "transformers_modeling_modules": [],
            "pretrained_model_objects": 0,
            "tokenizer_objects": 0,
            "optimizer_objects": 0,
            "output_inventory": [],
            "candidate_files": [],
            "checkpoint_files": [],
            "static_forbidden_model_call_sites": [],
            "observation_complete": True,
            "object_census_method": "gc_mro_scan_without_importing_model_tokenizer_or_optimizer_classes",
            "cuda_observation_method": "torch.cuda.is_initialized",
            "relevant_modules_absent_for_preimport_inference": False,
            "network_guard": hiter.NETWORK_GUARD_CONTRACT
            | {
                "installed": True,
                "wrappers_restored": True,
                "audit_hook_persistent": True,
                "attempt_count": 0,
            },
        },
        "completed_phase_records": phase_records,
        "final_terminal_publication": {
            "phase": "terminal_publication",
            "entered_ns_since_start": 20,
            "limit_ns": 60_000_000_000,
            "completion_observable_inside_terminal": False,
            "self_reference_boundary": "post_write_fsync_reopen_validation_and_process_exit_are_external_to_immutable_terminal_bytes",
        },
        "prepublication_elapsed_ns": 20,
        "partial_memory": {"expected_labels": hiter.memory_labels(), "rows": []},
        "full_freeze_failure_audit": {
            "head": execution,
            "tree": "3" * 40,
            "status": "",
            "plan_file_sha256": external_plan,
            "plan_sha256": plan["plan_sha256"],
            "plan_sidecar_sha256": hiter.sha256_bytes(f"{external_plan}\n".encode()),
            "plan_asset_hashes": plan["asset_sha256"],
            "errors": [],
        },
        "output_inventory_before_failure": [],
        "candidate_created": False,
        "checkpoint_created": False,
        "model_updated": False,
        "failure_sha256": "",
    }
    failure["failure_sha256"] = hiter.canonical_sha256(failure, omit="failure_sha256")
    validation_args = {
        "plan": plan,
        "expected_execution_commit": execution,
        "expected_plan_file_sha256": external_plan,
    }
    hiter.validate_failure(failure, **validation_args)
    stale = json_roundtrip(failure)
    stale["error"] = "tampered"
    with pytest.raises(hiter.ContractError, match="self hash"):
        hiter.validate_failure(stale, **validation_args)
    unsafe = json_roundtrip(failure)
    unsafe["actual_safety"]["cuda_initialized"] = True
    unsafe["failure_sha256"] = hiter.canonical_sha256(unsafe, omit="failure_sha256")
    with pytest.raises(hiter.ContractError, match="infrastructure"):
        hiter.validate_failure(unsafe, **validation_args)
    unexpected_output = json_roundtrip(failure)
    unexpected_output["actual_safety"]["output_inventory"] = ["intruder"]
    unexpected_output["output_inventory_before_failure"] = ["intruder"]
    unexpected_output["failure_sha256"] = hiter.canonical_sha256(
        unexpected_output, omit="failure_sha256"
    )
    with pytest.raises(hiter.ContractError, match="infrastructure"):
        hiter.validate_failure(unexpected_output, **validation_args)
    wrong_authority = json_roundtrip(failure)
    wrong_authority["execution_commit"] = "4" * 40
    wrong_authority["failure_sha256"] = hiter.canonical_sha256(wrong_authority, omit="failure_sha256")
    with pytest.raises(hiter.ContractError, match="launch authority"):
        hiter.validate_failure(wrong_authority, **validation_args)
    wrong_plan_authority = json_roundtrip(failure)
    wrong_plan_authority["authorized_plan_file_sha256"] = "5" * 64
    wrong_plan_authority["failure_sha256"] = hiter.canonical_sha256(
        wrong_plan_authority, omit="failure_sha256"
    )
    with pytest.raises(hiter.ContractError, match="plan authority"):
        hiter.validate_failure(wrong_plan_authority, **validation_args)
    malformed_memory = json_roundtrip(failure)
    malformed_memory["partial_memory"]["rows"] = [
        {"label": hiter.memory_labels()[0], "rss_bytes": True, "peak_rss_bytes": 1}
    ]
    malformed_memory["failure_sha256"] = hiter.canonical_sha256(malformed_memory, omit="failure_sha256")
    with pytest.raises(hiter.ContractError, match="memory value"):
        hiter.validate_failure(malformed_memory, **validation_args)


def json_roundtrip(value: object):
    return hiter.strict_json_loads(hiter.canonical_json(value))


def test_atomic_writer_refuses_nonempty_or_second_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner_path = ROOT / "scripts/latent/run_h_iter_phase0_generator_locality_v1.py"
    spec = importlib.util.spec_from_file_location("hiter_phase0_runner_test", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = tmp_path / hiter.RUN_IDENTITY
    monkeypatch.setitem(runner.RESOURCE_BOUNDS, "output_root", str(output))
    writer = runner.ArtifactWriter(output)
    reopened = writer.write("PROOF.json", {"x": 1}, 1024)
    assert reopened == b'{"x":1}\n'
    with pytest.raises(runner.InfrastructureInvalid, match="exclusive"):
        writer.write("FAILURE.json", {"x": 2}, 1024)

    second = tmp_path / "second"
    monkeypatch.setitem(runner.RESOURCE_BOUNDS, "output_root", str(second))
    second_writer = runner.ArtifactWriter(second)
    (second / "intruder").write_text("x")
    with pytest.raises(runner.InfrastructureInvalid, match="not empty"):
        second_writer.write("FAILURE.json", {"x": 2}, 1024)


def test_launcher_and_runtime_bind_shared_environment_and_outer_budget() -> None:
    launcher = (ROOT / "scripts/latent/run_h_iter_phase0_generator_locality_v1.sh").read_text()
    runner = (ROOT / "scripts/latent/run_h_iter_phase0_generator_locality_v1.py").read_text()
    assert hiter.EXPECTED_RUNTIME["shared_project_pyproject_sha256"] in launcher
    assert hiter.EXPECTED_RUNTIME["shared_project_uv_lock_sha256"] in launcher
    assert "timeout --signal=TERM --kill-after=60s 1200s" in launcher
    assert 'sys.executable != EXPECTED_RUNTIME["sys_executable"]' in runner
    assert 'sys.prefix != EXPECTED_RUNTIME["sys_prefix"]' in runner
    assert "class NetworkGuard" in runner and "sys.addaudithook" in runner
    assert 'tracker.enter("compute", RESOURCE_BOUNDS["compute_timeout_seconds"])' in runner
    assert 'tracker.enter("audit", RESOURCE_BOUNDS["audit_timeout_seconds"])' in runner
    assert 'tracker.enter("failure_audit", RESOURCE_BOUNDS["failure_audit_timeout_seconds"])' in runner


def test_network_guard_and_pre_torch_census_in_fresh_process(tmp_path: Path) -> None:
    runner_path = ROOT / "scripts/latent/run_h_iter_phase0_generator_locality_v1.py"
    program = f"""
import importlib.util
import socket
from pathlib import Path
spec = importlib.util.spec_from_file_location('hiter_guard_subprocess', {str(runner_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in module.sys.modules
output = Path({str(tmp_path)!r})
inventory = module.object_inventory(None, output)
assert inventory['relevant_modules_absent_for_preimport_inference'] is True
guard = module.NetworkGuard()
with guard:
    sock = socket.socket()
    operations = [
        lambda: sock.connect(('127.0.0.1', 9)),
        lambda: sock.connect_ex(('127.0.0.1', 9)),
        lambda: socket.create_connection(('127.0.0.1', 9)),
        lambda: socket.getaddrinfo('example.invalid', 443),
    ]
    for operation in operations:
        try:
            operation()
        except module.InfrastructureInvalid:
            pass
        else:
            raise AssertionError('network call was accepted')
    sock.close()
assert guard.installed is True and guard.wrappers_restored is True
assert guard.attempt_count == 4
"""
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_proof_validator_binds_launch_authority_before_payload_details() -> None:
    plan = load_asset("phase0-plan.json")
    plan["runtime"] = hiter.EXPECTED_RUNTIME
    plan["resource_bounds"] = hiter.RESOURCE_BOUNDS
    plan["safety_boundary"] = hiter.DECISION_BOUNDARY | {
        "coordinator_e33_loaded": False,
        "worker_h176_loaded": False,
        "tokenizer_calls": 0,
        "model_forwards": 0,
        "model_backwards": 0,
        "optimizer_steps": 0,
        "synthetic_cpu_backwards": 40,
        "transformers_modeling_imports": 0,
        "network_guard": hiter.NETWORK_GUARD_CONTRACT,
    }
    plan["plan_sha256"] = hiter.canonical_sha256(plan, omit="plan_sha256")
    proof = {
        key: None
        for key in {
            "schema_version", "status", "mechanism", "run_identity", "execution_commit",
            "mechanism_code_commit", "plan_file_sha256", "plan_sha256", "runtime", "asset_audit",
            "banks", "structural_audit", "overlap_audit", "operation_schedule", "locality",
            "tamper_audit", "counts", "safety", "resources", "memory", "full_freeze",
            "decision_boundary", "proof_sha256",
        }
    }
    proof.update(
        {
            "schema_version": hiter.PROOF_SCHEMA,
            "status": "h_iter_phase0_generator_locality_validated",
            "mechanism": hiter.MECHANISM,
            "run_identity": hiter.RUN_IDENTITY,
            "execution_commit": "3" * 40,
            "mechanism_code_commit": plan["mechanism_code_commit"],
            "plan_file_sha256": "2" * 64,
            "plan_sha256": plan["plan_sha256"],
        }
    )
    with pytest.raises(hiter.ContractError, match="launch authority"):
        hiter.validate_proof(
            proof,
            plan=plan,
            banks={},
            selection={},
            schedule={},
            overlap={},
            expected_execution_commit="1" * 40,
            expected_plan_file_sha256="2" * 64,
        )
