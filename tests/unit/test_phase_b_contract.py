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
    receipt["receipt_sha256"] = canonical_json_sha256(receipt, omitted_fields=("receipt_sha256",))
    receipt_path = (tmp_path / "A0C_SUCCESS.json").resolve()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    binding = {
        "schema_version": "q35-2b-phase-b-a0c-binding/v1",
        "status": "bound",
        "required_claim": "four_probe_carrier_only_for_phase_b",
        "receipt_absolute_path": str(receipt_path),
        "receipt_file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "receipt_canonical_sha256": receipt["receipt_sha256"],
        "receipt_schema_version": receipt["schema_version"],
        "a0c_plan_sha256": "a" * 64,
        "a0c_execution_commit": "b" * 40,
        "identity": {},
        "predicate_paths": {
            name: {"path": f"probes.*.{name}", "expected": [True, True, True, True]} for name in REQUIRED_A0C_PREDICATES
        },
    }
    binding["predicate_paths"]["four_probes_completed"] = {"path": "complete_probes", "expected": 4}
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
        "complete_probes": 4,
        "probes": [{name: True for name in REQUIRED_A0C_PREDICATES} for _ in range(4)],
    }
    binding_path, hash_path = _write_binding(tmp_path, receipt=receipt)

    result = validate_a0c_binding(binding_path, hash_path)

    assert result.receipt == receipt
    assert result.receipt_canonical_sha256 == canonical_json_sha256(receipt, omitted_fields=("receipt_sha256",))


def test_binding_rejects_wrong_a0c_internal_hash_even_when_file_hash_is_bound(tmp_path: Path) -> None:
    receipt = {
        "schema_version": "q35-2b-a0c-carrier-receipt/v1",
        "status": "SUCCESS",
        "claim": "four_probe_carrier_only_for_phase_b",
        "complete_probes": 4,
        "probes": [{name: True for name in REQUIRED_A0C_PREDICATES} for _ in range(4)],
    }
    binding_path, hash_path = _write_binding(tmp_path, receipt=receipt)
    receipt_path = tmp_path / "A0C_SUCCESS.json"
    receipt["receipt_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["receipt_file_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    binding_hash = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    hash_path.write_text(f"{binding_hash}  {binding_path.name}\n", encoding="ascii")

    with pytest.raises(PhaseBContractError, match="internal canonical"):
        validate_a0c_binding(binding_path, hash_path)


def test_binding_rejects_receipt_changed_after_freeze(tmp_path: Path) -> None:
    receipt = {
        "schema_version": "q35-2b-a0c-carrier-receipt/v1",
        "status": "SUCCESS",
        "claim": "four_probe_carrier_only_for_phase_b",
        "complete_probes": 4,
        "probes": [{name: True for name in REQUIRED_A0C_PREDICATES} for _ in range(4)],
    }
    binding_path, hash_path = _write_binding(tmp_path, receipt=receipt)
    receipt_path = tmp_path / "A0C_SUCCESS.json"
    receipt["complete_probes"] = 3
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(PhaseBContractError, match="file hash"):
        validate_a0c_binding(binding_path, hash_path)


def test_binding_requires_every_named_carrier_predicate(tmp_path: Path) -> None:
    receipt = {
        "schema_version": "q35-2b-a0c-carrier-receipt/v1",
        "status": "SUCCESS",
        "claim": "four_probe_carrier_only_for_phase_b",
        "complete_probes": 4,
        "probes": [{name: True for name in REQUIRED_A0C_PREDICATES} for _ in range(4)],
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
    runner.PLAN_SHA256 = hashlib.sha256(runner.PLAN.read_bytes()).hexdigest()
    args.plan = runner.PLAN
    args.selection = runner.SELECTION

    with pytest.raises(PhaseBContractError, match="not prospectively"):
        runner.preflight_before_heavy_imports(args)
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert not args.output_dir.exists()


def test_checked_in_binding_maps_and_validates_exact_a0c_carrier_receipt(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    experiment = repository / "experiments/qwen35-2b-latent-coordinator-v1"
    binding = json.loads((experiment / "phase-b-a0c-binding-v1.json").read_text(encoding="utf-8"))
    receipt_source = experiment / "a0c-carrier-success-receipt-v1.json"
    receipt_path = (tmp_path / "receipt.json").resolve()
    receipt_path.write_bytes(receipt_source.read_bytes())
    binding["receipt_absolute_path"] = str(receipt_path)
    binding_path = (tmp_path / "binding.json").resolve()
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    binding_hash = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    hash_path = (tmp_path / "binding.sha256").resolve()
    hash_path.write_text(f"{binding_hash}  {binding_path.name}\n", encoding="ascii")

    validated = validate_a0c_binding(binding_path, hash_path)

    assert binding["status"] == "bound"
    assert set(binding["predicate_paths"]) == set(REQUIRED_A0C_PREDICATES)
    assert validated.receipt_file_sha256 == "d88dd97eb37c9c3dd61bc07fe422df6c7fa0034837897346e43ed16bb634e63c"
    assert validated.receipt_canonical_sha256 == "40dde68d34deb592f864739b48da8c22faafe470f2f0bc6708bce608ae482de7"


def test_failure_audit_rehashes_e33_binding_receipt_and_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    receipt = {
        "schema_version": "q35-2b-a0c-carrier-receipt/v1",
        "status": "SUCCESS",
        "claim": "four_probe_carrier_only_for_phase_b",
        "complete_probes": 4,
        "probes": [{name: True for name in REQUIRED_A0C_PREDICATES} for _ in range(4)],
    }
    binding_path, hash_path = _write_binding(tmp_path, receipt=receipt)
    binding = validate_a0c_binding(binding_path, hash_path)
    model_path = tmp_path / "e33"
    model_path.mkdir()
    weight = model_path / "model.safetensors"
    weight.write_bytes(b"frozen-e33")
    metadata_names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
    )
    for name in metadata_names:
        (model_path / name).write_text(name, encoding="utf-8")
    plan_file = (tmp_path / "plan.json").resolve()
    selection_file = (tmp_path / "selection.json").resolve()
    plan_file.write_text("{}", encoding="utf-8")
    selection_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "PLAN", plan_file)
    monkeypatch.setattr(runner, "SELECTION", selection_file)
    monkeypatch.setattr(runner, "PLAN_SHA256", hashlib.sha256(plan_file.read_bytes()).hexdigest())
    monkeypatch.setattr(runner, "SELECTION_SHA256", hashlib.sha256(selection_file.read_bytes()).hexdigest())
    plan = {
        "protected_models": {
            "coordinator": {
                "host_path": str(model_path.resolve()),
                "expected_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            }
        },
        "model_metadata_sha256": {
            name: hashlib.sha256((model_path / name).read_bytes()).hexdigest() for name in metadata_names
        },
    }

    audit = runner._post_failure_hash_audit(plan, binding)

    assert audit["audit_complete"] is True
    assert audit["hash_probe_error"] is None
    assert audit["e33"]["weight_file_matches"] is True
    assert audit["e33"]["metadata_matches"] is True
    assert audit["a0c"]["binding_matches"] is True
    assert audit["a0c"]["receipt_file_matches"] is True
    assert audit["a0c"]["receipt_internal_matches"] is True

    weight.unlink()
    (tmp_path / "A0C_SUCCESS.json").write_text("{}", encoding="utf-8")
    failed_audit = runner._post_failure_hash_audit(plan, binding)
    assert failed_audit["audit_complete"] is False
    assert "e33:" in failed_audit["hash_probe_error"]
    assert "a0c:" in failed_audit["hash_probe_error"]


def test_failure_classification_and_cleanup_headroom_are_explicit() -> None:
    runner = _load_runner()

    assert runner._classify_failure(runner.PhaseBMechanismRejected("predicate")) == "mechanism_rejected"
    assert runner._classify_failure(runner.PhaseBWallClockExceeded("timeout")) == "infrastructure_invalid"
    assert runner._classify_failure(MemoryError("oom")) == "infrastructure_invalid"
    assert runner.COMPUTE_LIMIT_SECONDS + runner.FAILURE_AUDIT_HEADROOM_SECONDS == runner.WALL_CLOCK_LIMIT_SECONDS
    assert runner.FAILURE_AUDIT_HEADROOM_SECONDS == 300


def test_launcher_supplies_independent_outer_timeout_and_exact_environment() -> None:
    repository = Path(__file__).resolve().parents[2]
    launcher = (repository / "scripts/latent/run_phase_b_fixed_depth_smoke_a0c_v1.sh").read_text(encoding="utf-8")

    assert "timeout --signal=TERM --kill-after=30s 120m" in launcher
    assert "UV_PROJECT_ENVIRONMENT=\"$SHARED_ENV\"" in launcher
    assert "CUDA_VISIBLE_DEVICES=0,1" in launcher
    assert "--no-sync python" in launcher
    assert "--execution-commit \"$EXECUTION_COMMIT\"" in launcher


def _load_runner():
    repository = Path(__file__).resolve().parents[2]
    runner_path = repository / "scripts/latent/run_phase_b_fixed_depth_smoke_v1.py"
    spec = importlib.util.spec_from_file_location(f"phase_b_runner_{id(runner_path)}", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner
