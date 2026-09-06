from __future__ import annotations

import ast
import subprocess
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
    assert "state[start_index]" in readout_text


def test_tamper_schedule_and_memory_labels_are_stable() -> None:
    tamper = load_asset("tamper-schedule.json")
    assert tamper == hiter.build_tamper_schedule()
    assert [row["name"] for row in tamper["mechanism_tampers"]] == hiter.MECHANISM_TAMPERS
    assert [row["name"] for row in tamper["receipt_tampers"]] == hiter.RECEIPT_TAMPERS
    labels = hiter.memory_labels()
    assert len(labels) == len(set(labels)) == 26
