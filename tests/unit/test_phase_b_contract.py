from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from prime_rl.phase_b_contract import (
    REQUIRED_A0C_PREDICATES,
    PhaseBContractError,
    atomic_exclusive_json,
    canonical_json_sha256,
    normalize_assistant_tool_call_arguments,
    validate_a0c_binding,
    validate_failed_start_evidence,
    validate_plan_authorization,
)

EXACT_ACTIONS_BY_TASK_KEY = {
    "document_adaptive_d0-v0-i34100:solve-anchor-1": "solve_owned",
    "document_adaptive_d2-v0-i34100:manager": "delegate_terminal",
    "document_adaptive_d3-v0-i34100:top-manager": "delegate_coordinator",
    "document_adaptive_d0-v0-i34100:solve-anchor-2": "solve_owned",
    "document_adaptive_d2-v1-i34100:manager": "delegate_terminal",
    "document_adaptive_d3-v0-i34101:top-manager": "delegate_coordinator",
    "document_adaptive_d0-v1-i34100:solve-anchor-1": "solve_owned",
    "document_adaptive_d2-v2-i34100:manager": "delegate_terminal",
    "document_adaptive_d3-v1-i34100:top-manager": "delegate_coordinator",
    "document_adaptive_d0-v1-i34100:solve-anchor-2": "solve_owned",
    "document_adaptive_d2-v3-i34100:manager": "delegate_terminal",
    "document_adaptive_d3-v1-i34101:top-manager": "delegate_coordinator",
}


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


def test_exact_twelve_targets_normalize_only_arguments_and_preserve_actions() -> None:
    observed_actions: list[str] = []
    for task_key, action in EXACT_ACTIONS_BY_TASK_KEY.items():
        raw_arguments = json.dumps({"action": action}, separators=(",", ":"))
        messages = [
            {"role": "system", "content": "contract"},
            {"role": "user", "content": task_key},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{task_key}",
                        "type": "function",
                        "function": {"name": "select_cognitive_action", "arguments": raw_arguments},
                    }
                ],
            },
        ]
        original = json.loads(json.dumps(messages))

        normalized, records = normalize_assistant_tool_call_arguments(messages, expected_action=action)

        assert messages == original
        assert len(records) == 1
        assert records[0]["modified_path"] == "messages.2.tool_calls.0.function.arguments"
        assert records[0]["source_kind"] == "json_string"
        assert records[0]["raw_arguments_sha256"] == hashlib.sha256(raw_arguments.encode()).hexdigest()
        assert records[0]["normalized_arguments_sha256"] == canonical_json_sha256({"action": action})
        assert records[0]["normalized_arguments"] == {"action": action}
        assert normalized[-1]["tool_calls"][0]["function"]["arguments"] == {"action": action}
        assert normalized[-1]["tool_calls"][0]["id"] == original[-1]["tool_calls"][0]["id"]
        restored = json.loads(json.dumps(normalized))
        restored[-1]["tool_calls"][0]["function"]["arguments"] = raw_arguments
        assert restored == original
        observed_actions.append(action)
    assert {action: observed_actions.count(action) for action in set(observed_actions)} == {
        "solve_owned": 4,
        "delegate_terminal": 4,
        "delegate_coordinator": 4,
    }


@pytest.mark.parametrize("raw", ["[]", "42", '"text"', "null", "{bad", '{"action":NaN}', '{"a":1,"a":2}'])
def test_tool_argument_normalization_rejects_nonobject_or_malformed_json(raw: str) -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "select_cognitive_action", "arguments": raw}}],
        }
    ]

    with pytest.raises(PhaseBContractError):
        normalize_assistant_tool_call_arguments(messages, expected_action="solve_owned")


@pytest.mark.parametrize(
    ("messages", "expected_action"),
    [
        (
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "wrong", "arguments": '{"action":"solve_owned"}'}}],
                }
            ],
            "solve_owned",
        ),
        (
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "select_cognitive_action",
                                "arguments": '{"action":"solve_owned"}',
                            }
                        },
                        {
                            "function": {
                                "name": "select_cognitive_action",
                                "arguments": '{"action":"solve_owned"}',
                            }
                        },
                    ],
                }
            ],
            "solve_owned",
        ),
        (
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "select_cognitive_action",
                                "arguments": {"action": "solve_owned"},
                            }
                        }
                    ],
                }
            ],
            "solve_owned",
        ),
        (
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "select_cognitive_action",
                                "arguments": '{"action":"delegate_terminal"}',
                            }
                        }
                    ],
                }
            ],
            "solve_owned",
        ),
        (
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "select_cognitive_action",
                                "arguments": '{"action":"solve_owned","extra":true}',
                            }
                        }
                    ],
                }
            ],
            "solve_owned",
        ),
        (
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "select_cognitive_action",
                                "arguments": '{"action":"solve_owned"}',
                            }
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "select_cognitive_action",
                                "arguments": '{"action":"solve_owned"}',
                            }
                        }
                    ],
                },
            ],
            "solve_owned",
        ),
    ],
)
def test_tool_argument_normalization_rejects_scope_or_semantic_drift(
    messages: list[dict[str, object]], expected_action: str
) -> None:
    original = json.loads(json.dumps(messages))

    with pytest.raises(PhaseBContractError):
        normalize_assistant_tool_call_arguments(messages, expected_action=expected_action)

    assert messages == original


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


def test_failed_start_binding_validates_exact_artifacts_and_predicates(tmp_path: Path) -> None:
    failure = {
        "status": "infrastructure_invalid",
        "error_type": "TypeError",
        "error": "Can only get item pairs from a mapping.",
        "post_failure_hash_audit": {"audit_complete": True, "hash_probe_error": None},
    }
    failure_path = (tmp_path / "FAILURE.json").resolve()
    log_path = (tmp_path / "run.log").resolve()
    failure_path.write_text(json.dumps(failure), encoding="utf-8")
    log_path.write_text("weights loaded\nPhase B failed: TypeError: Can only get item pairs from a mapping.\n")
    binding = {
        "schema_version": "q35-2b-phase-b-failed-start-binding/v1",
        "status": "bound_infrastructure_invalid",
        "failure_absolute_path": str(failure_path),
        "failure_file_sha256": hashlib.sha256(failure_path.read_bytes()).hexdigest(),
        "log_absolute_path": str(log_path),
        "log_file_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "failure_predicates": {
            "status": {"path": "status", "expected": "infrastructure_invalid"},
            "audit": {"path": "post_failure_hash_audit.audit_complete", "expected": True},
        },
        "required_log_markers": ["Phase B failed: TypeError"],
    }
    binding_path = (tmp_path / "binding.json").resolve()
    hash_path = (tmp_path / "binding.sha256").resolve()
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    binding_hash = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    hash_path.write_text(f"{binding_hash}  {binding_path.name}\n", encoding="ascii")

    validated = validate_failed_start_evidence(binding_path, hash_path)

    assert validated.failure == failure
    assert validated.failure_file_sha256 == binding["failure_file_sha256"]
    assert validated.log_file_sha256 == binding["log_file_sha256"]

    failure["status"] = "mechanism_rejected"
    failure_path.write_text(json.dumps(failure), encoding="utf-8")
    with pytest.raises(PhaseBContractError, match="FAILURE hash"):
        validate_failed_start_evidence(binding_path, hash_path)


def test_checked_in_failed_start_assets_are_exact_and_prove_no_update_boundary(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    experiment = repository / "experiments/qwen35-2b-latent-coordinator-v1"
    failure_source = experiment / "phase-b-fixed-depth-smoke-a0c-v1-FAILURE.json"
    log_source = experiment / "phase-b-fixed-depth-smoke-a0c-v1-run.log"
    binding_source = experiment / "phase-b-a0c-v1-failed-start-binding.json"
    binding = json.loads(binding_source.read_text(encoding="utf-8"))
    assert hashlib.sha256(failure_source.read_bytes()).hexdigest() == (
        "0ddc8b349de0497f5c886cbff85d6e12b9a7a0156c16c2789b9d5985ded1d014"
    )
    assert hashlib.sha256(log_source.read_bytes()).hexdigest() == (
        "4bb4d7941a5207d4384230d0873b6df2d2cafd71c3c74fbc2f31dab9ee05ca6b"
    )
    assert hashlib.sha256(binding_source.read_bytes()).hexdigest() == (
        experiment / "phase-b-a0c-v1-failed-start-binding.sha256"
    ).read_text(encoding="ascii").split()[0]

    binding["failure_absolute_path"] = str(failure_source.resolve())
    binding["log_absolute_path"] = str(log_source.resolve())
    binding_path = (tmp_path / "binding.json").resolve()
    hash_path = (tmp_path / "binding.sha256").resolve()
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    binding_hash = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    hash_path.write_text(f"{binding_hash}  {binding_path.name}\n", encoding="ascii")

    validated = validate_failed_start_evidence(binding_path, hash_path)

    control = validated.binding["control_flow_proof"]
    assert control["optimizer_construction_present"] is False
    assert control["optimizer_step_present"] is False
    assert control["checkpoint_write_present"] is False
    assert control["generation_present"] is False
    prior_source = subprocess.run(
        [
            "git",
            "show",
            f"{control['prior_execution_commit']}:{control['prior_runner_path']}",
        ],
        cwd=repository,
        capture_output=True,
        check=True,
    ).stdout
    assert hashlib.sha256(prior_source).hexdigest() == control["prior_runner_sha256"]


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


def test_repair_plan_is_exact_and_prior_plans_remain_immutable() -> None:
    repository = Path(__file__).resolve().parents[2]
    experiment = repository / "experiments/qwen35-2b-latent-coordinator-v1"
    old_plan = json.loads((experiment / "phase-b-fixed-depth-smoke-v1-plan.json").read_text(encoding="utf-8"))
    plan_path = experiment / "phase-b-fixed-depth-smoke-a0c-v1-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    repair_plan_path = experiment / "phase-b-fixed-depth-smoke-a0c-br1-plan.json"
    repair_plan = json.loads(repair_plan_path.read_text(encoding="utf-8"))
    selection_path = experiment / "phase-b-fixed-depth-smoke-v1-selection.json"

    with pytest.raises(PhaseBContractError):
        validate_plan_authorization(old_plan)
    validate_plan_authorization(plan)
    assert plan["implementation_commit"] == "476171123f642f8c7857606bd97d32ad93721eaa"
    assert set(plan["arms"]) == {"BASE", "STATIC", "FFN", "RECURRENT"}
    assert plan["arms"]["RECURRENT"]["local_depth"] == 4
    assert plan["matched_conditions"]["optimizer_updates"] == 0
    assert plan["matched_conditions"]["model_generation"] is False
    assert plan["failure_classification"]["compute_limit_seconds"] == 6840
    assert plan["failure_classification"]["failure_audit_headroom_seconds"] == 300
    assert plan["failure_classification"]["terminal_publication_headroom_seconds"] == 60
    assert plan["failure_classification"]["outer_wall_clock_seconds"] == 7200
    assert plan["a0c_dependency"]["a0r_rejection_preserved"]["status"] == "mechanism_rejected"
    assert Path(plan["a0c_dependency"]["binding_path"]).parent != Path(plan["outputs"]["directory"])
    assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == plan["data"]["selection_sha256"]

    runner = _load_runner()
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == (
        "154f65eead8d206316e111b8e18dc3f86fe4ea69a4f0d731de8390749d800e1b"
    )
    validate_plan_authorization(repair_plan)
    assert repair_plan["implementation_commit"] == "c4ba6c59c03780d5aa4e9a3430be4f56755a3381"
    assert repair_plan["arms"] == plan["arms"]
    assert repair_plan["matched_conditions"] == plan["matched_conditions"]
    assert repair_plan["data"] == plan["data"]
    assert repair_plan["repair_dependency"]["failure_file_sha256"] == (
        "0ddc8b349de0497f5c886cbff85d6e12b9a7a0156c16c2789b9d5985ded1d014"
    )
    assert repair_plan["repair_dependency"]["log_file_sha256"] == (
        "4bb4d7941a5207d4384230d0873b6df2d2cafd71c3c74fbc2f31dab9ee05ca6b"
    )
    assert repair_plan["repair_dependency"]["prior_output_reuse_forbidden"] is True
    assert repair_plan["outputs"]["directory"] != plan["outputs"]["directory"]
    assert repair_plan["tokenizer_only_repair_preflight"]["rows"] == 12
    assert runner.PLAN.name == "phase-b-fixed-depth-smoke-a0c-br1-plan.json"
    assert runner.PLAN_SHA256 == hashlib.sha256(repair_plan_path.read_bytes()).hexdigest()


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
    assert (
        runner.COMPUTE_LIMIT_SECONDS
        + runner.FAILURE_AUDIT_HEADROOM_SECONDS
        + runner.TERMINAL_PUBLICATION_HEADROOM_SECONDS
        == runner.WALL_CLOCK_LIMIT_SECONDS
    )
    assert runner.FAILURE_AUDIT_HEADROOM_SECONDS == 300
    assert runner.TERMINAL_PUBLICATION_HEADROOM_SECONDS == 60


def test_preflight_only_returns_without_heavy_import_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = _load_runner()
    output = (tmp_path / "never-created").resolve()
    args = runner.argparse.Namespace(preflight_only=True, execution_commit="a" * 40, output_dir=output)
    binding = SimpleNamespace(
        binding_file_sha256="b" * 64,
        receipt_file_sha256="c" * 64,
        receipt_canonical_sha256="d" * 64,
    )
    failed_start = SimpleNamespace(binding_file_sha256="e" * 64)
    fake_parquet = ModuleType("pyarrow.parquet")
    fake_pyarrow = ModuleType("pyarrow")
    fake_pyarrow.__path__ = []  # type: ignore[attr-defined]
    fake_pyarrow.parquet = fake_parquet  # type: ignore[attr-defined]
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "preflight_before_heavy_imports", lambda _args: ({}, {}, binding, failed_start))
    monkeypatch.setattr(
        runner,
        "_tokenizer_only_preflight",
        lambda **_kwargs: {"proofs": [{"task_key": key} for key in EXACT_ACTIONS_BY_TASK_KEY], "cuda_initialized": False},
    )

    assert runner.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "tokenizer_preflight_only_passed"
    assert len(report["normalization_and_render_proofs"]) == 12
    assert report["model_loaded"] is False
    assert report["cuda_initialized_during_preflight"] is False
    assert report["output_created"] is False
    assert "torch" not in sys.modules
    assert not output.exists()


def test_tokenizer_preflight_normalizes_and_renders_all_twelve_before_model_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    rows = []
    for task_key, action in EXACT_ACTIONS_BY_TASK_KEY.items():
        rows.append(
            {
                "task_key": task_key,
                "action": action,
                "messages": [
                    {"role": "system", "content": "contract"},
                    {"role": "user", "content": task_key},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call-{task_key}",
                                "type": "function",
                                "function": {
                                    "name": "select_cognitive_action",
                                    "arguments": json.dumps({"action": action}, separators=(",", ":")),
                                },
                            }
                        ],
                    },
                ],
                "tools": [{"type": "function", "function": {"name": "select_cognitive_action"}}],
            }
        )
    source_snapshot = json.loads(json.dumps(rows))

    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, object]],
            *,
            tools: list[dict[str, object]],
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert tools and tokenize is False
            if add_generation_prompt:
                return json.dumps(messages, sort_keys=True) + "<assistant>"
            if messages[-1]["role"] == "assistant":
                arguments = messages[-1]["tool_calls"][0]["function"]["arguments"]  # type: ignore[index]
                assert isinstance(arguments, dict)
                return json.dumps(messages[:-1], sort_keys=True) + "<assistant>" + json.dumps(
                    messages[-1], sort_keys=True
                )
            return json.dumps(messages, sort_keys=True)

        def __call__(self, rendered: str, *, add_special_tokens: bool) -> SimpleNamespace:
            assert add_special_tokens is False
            return SimpleNamespace(input_ids=[value + 1 for value in rendered.encode()])

    class FakeTable:
        def to_pylist(self) -> list[dict[str, object]]:
            return rows

    fake_parquet = SimpleNamespace(read_table=lambda _path: FakeTable())
    fake_auto_tokenizer = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeTokenizer())
    monkeypatch.setattr(runner, "_validate_transformers_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_model_file", lambda _path: tmp_path / "model.safetensors")
    monkeypatch.setattr(runner, "_metadata_hashes", lambda _path: {"config.json": "metadata"})

    hashes = {
        "model.safetensors": "model",
        "train.parquet": "parquet",
        "MANIFEST.json": "manifest",
    }
    monkeypatch.setattr(runner, "file_sha256", lambda path: hashes[path.name])
    plan = {
        "protected_models": {"coordinator": {"host_path": str(tmp_path), "expected_sha256": "model"}},
        "model_metadata_sha256": {"config.json": "metadata"},
        "data": {
            "host_parquet_path": str(tmp_path / "train.parquet"),
            "host_manifest_path": str(tmp_path / "MANIFEST.json"),
            "source_parquet_sha256": "parquet",
            "source_manifest_sha256": "manifest",
        },
    }
    selection = {
        "probe_task_keys": list(EXACT_ACTIONS_BY_TASK_KEY),
        "backward_probe_task_key": "document_adaptive_d3-v0-i34100:top-manager",
    }

    context = runner._tokenizer_only_preflight(
        plan=plan,
        selection=selection,
        parquet=fake_parquet,
        transformers=SimpleNamespace(),
        AutoTokenizer=fake_auto_tokenizer,
    )

    assert rows == source_snapshot
    assert len(context["rows"]) == len(context["proofs"]) == 12
    assert context["cuda_initialized"] is False
    for proof in context["proofs"]:
        assert proof["modified_paths"] == ["messages.2.tool_calls.0.function.arguments"]
        assert proof["normalized_arguments"] == {"action": proof["action"]}
        assert proof["full_target_tokens"] > proof["generation_prefix_tokens"] > proof["plain_prefix_tokens"]


def test_launcher_supplies_independent_outer_timeout_and_exact_environment() -> None:
    repository = Path(__file__).resolve().parents[2]
    launcher = (repository / "scripts/latent/run_phase_b_fixed_depth_smoke_a0c_br1.sh").read_text(encoding="utf-8")

    assert "timeout --signal=TERM --kill-after=30s 120m" in launcher
    assert 'UV_PROJECT_ENVIRONMENT="$SHARED_ENV"' in launcher
    assert "CUDA_VISIBLE_DEVICES=0,1" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "ROOT_AUTHORIZED_PLAN_SHA256=$2" in launcher
    assert "sha256sum --check phase-b-fixed-depth-smoke-a0c-br1.sha256" in launcher
    assert "[--preflight-only]" in launcher
    assert "--no-sync python" in launcher
    assert '--execution-commit "$EXECUTION_COMMIT"' in launcher


def test_repair_freeze_manifest_closes_exact_launch_assets() -> None:
    repository = Path(__file__).resolve().parents[2]
    experiment = repository / "experiments/qwen35-2b-latent-coordinator-v1"

    result = subprocess.run(
        ["sha256sum", "--check", "phase-b-fixed-depth-smoke-a0c-br1.sha256"],
        cwd=experiment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "phase-b-fixed-depth-smoke-a0c-v1-FAILURE.json: OK" in result.stdout
    assert "phase-b-fixed-depth-smoke-a0c-v1-run.log: OK" in result.stdout
    assert "../../scripts/latent/run_phase_b_fixed_depth_smoke_v1.py: OK" in result.stdout


def _load_runner():
    repository = Path(__file__).resolve().parents[2]
    runner_path = repository / "scripts/latent/run_phase_b_fixed_depth_smoke_v1.py"
    spec = importlib.util.spec_from_file_location(f"phase_b_runner_{id(runner_path)}", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner
