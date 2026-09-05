from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from prime_rl.phase_b_contract import (
    REQUIRED_A0C_PREDICATES,
    PhaseBContractError,
    atomic_exclusive_json,
    canonical_json_sha256,
    validate_a0c_binding,
    validate_plan_authorization,
)


def _write_binding(tmp_path: Path, *, receipt: dict[str, object]) -> tuple[Path, Path]:
    receipt_path = (tmp_path / "A0C_SUCCESS.json").resolve()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    binding = {
        "schema_version": "q35-2b-phase-b-a0c-binding/v1",
        "status": "bound",
        "required_claim": "four_probe_carrier_only_for_phase_b",
        "receipt_absolute_path": str(receipt_path),
        "receipt_file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "receipt_canonical_sha256": canonical_json_sha256(receipt),
        "receipt_schema_version": receipt["schema_version"],
        "a0c_plan_sha256": "a" * 64,
        "a0c_execution_commit": "b" * 40,
        "identity": {},
        "predicate_paths": {
            name: {"path": f"predicates.{name}", "expected": receipt["predicates"][name]}  # type: ignore[index]
            for name in REQUIRED_A0C_PREDICATES
        },
    }
    binding_path = (tmp_path / "A0C_BINDING.json").resolve()
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    binding_hash = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    hash_path = (tmp_path / "A0C_BINDING.sha256").resolve()
    hash_path.write_text(f"{binding_hash}  {binding_path.name}\n", encoding="ascii")
    return binding_path, hash_path


def test_unresolved_binding_and_old_plan_are_non_launchable(tmp_path: Path) -> None:
    binding_path = (tmp_path / "A0C_BINDING.json").resolve()
    binding_path.write_text(
        json.dumps(
            {
                "schema_version": "q35-2b-phase-b-a0c-binding/v1",
                "status": "unresolved",
                "required_claim": "four_probe_carrier_only_for_phase_b",
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    hash_path = (tmp_path / "A0C_BINDING.sha256").resolve()
    hash_path.write_text(f"{digest}  {binding_path.name}\n", encoding="ascii")

    with pytest.raises(PhaseBContractError, match="unresolved"):
        validate_a0c_binding(binding_path, hash_path)
    with pytest.raises(PhaseBContractError, match="not prospectively"):
        validate_plan_authorization({"status": "frozen_pending_a0_receipt_binding_not_authorized"})


def test_exact_bound_receipt_and_predicates_validate(tmp_path: Path) -> None:
    receipt = {
        "schema_version": "q35-2b-a0c-carrier-receipt/v1",
        "status": "SUCCESS",
        "claim": "four_probe_carrier_only_for_phase_b",
        "predicates": {name: 4 if name == "four_probes_completed" else True for name in REQUIRED_A0C_PREDICATES},
    }
    binding_path, hash_path = _write_binding(tmp_path, receipt=receipt)

    result = validate_a0c_binding(binding_path, hash_path)

    assert result.receipt == receipt
    assert result.receipt_canonical_sha256 == canonical_json_sha256(receipt)


def test_binding_rejects_receipt_changed_after_freeze(tmp_path: Path) -> None:
    receipt = {
        "schema_version": "q35-2b-a0c-carrier-receipt/v1",
        "status": "SUCCESS",
        "claim": "four_probe_carrier_only_for_phase_b",
        "predicates": {name: 4 if name == "four_probes_completed" else True for name in REQUIRED_A0C_PREDICATES},
    }
    binding_path, hash_path = _write_binding(tmp_path, receipt=receipt)
    receipt_path = tmp_path / "A0C_SUCCESS.json"
    receipt["predicates"]["four_probes_completed"] = 3  # type: ignore[index]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(PhaseBContractError, match="file hash"):
        validate_a0c_binding(binding_path, hash_path)


def test_binding_requires_every_named_carrier_predicate(tmp_path: Path) -> None:
    receipt = {
        "schema_version": "q35-2b-a0c-carrier-receipt/v1",
        "status": "SUCCESS",
        "claim": "four_probe_carrier_only_for_phase_b",
        "predicates": {name: 4 if name == "four_probes_completed" else True for name in REQUIRED_A0C_PREDICATES},
    }
    binding_path, hash_path = _write_binding(tmp_path, receipt=receipt)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    del binding["predicate_paths"]["capture_detached_all"]
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    digest = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    hash_path.write_text(f"{digest}  {binding_path.name}\n", encoding="ascii")

    with pytest.raises(PhaseBContractError, match="exact required"):
        validate_a0c_binding(binding_path, hash_path)


def test_atomic_terminal_receipt_is_exclusive(tmp_path: Path) -> None:
    output = (tmp_path / "run").resolve()
    output.mkdir()
    atomic_exclusive_json(output, "SUCCESS.json", {"status": "SUCCESS"}, maximum_directory_bytes=1024)

    with pytest.raises(FileExistsError, match="terminal receipt"):
        atomic_exclusive_json(output, "FAILURE.json", {"status": "FAILURE"}, maximum_directory_bytes=1024)
    assert [path.name for path in output.iterdir()] == ["SUCCESS.json"]


def test_checked_in_runner_refuses_old_plan_before_heavy_imports(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    runner_path = repository / "scripts/latent/run_phase_b_fixed_depth_smoke_v1.py"
    spec = importlib.util.spec_from_file_location("phase_b_runner_under_test", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    experiment = repository / "experiments/qwen35-2b-latent-coordinator-v1"
    args = runner.argparse.Namespace(
        plan=runner.PLAN,
        selection=runner.SELECTION,
        a0c_binding=(tmp_path / "must-not-be-read.json").resolve(),
        a0c_binding_hash=(tmp_path / "must-not-be-read.sha256").resolve(),
        output_dir=(tmp_path / "must-not-be-created").resolve(),
    )

    runner.PLAN = (experiment / "phase-b-fixed-depth-smoke-v1-plan.json").resolve()
    runner.SELECTION = (experiment / "phase-b-fixed-depth-smoke-v1-selection.json").resolve()
    args.plan = runner.PLAN
    args.selection = runner.SELECTION

    with pytest.raises(PhaseBContractError, match="not prospectively"):
        runner.preflight_before_heavy_imports(args)
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert not args.output_dir.exists()
