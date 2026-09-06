import base64
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from prime_rl.phase_b_contract import PhaseBContractError, canonical_json_sha256
from prime_rl.phase_b_ipc1 import (
    ACTIONS,
    EVALUATION_DEPTHS,
    FAILURE_STATUS_CLASSES,
    SELECTIONS,
    build_cache_guard_labels,
    build_memory_checkpoint_labels,
    build_model_call_schedule,
    canonical_plan_sha256,
    canonical_terminal_bytes,
    evaluate_common_arm,
    evaluate_recurrent_value,
    roundtrip_validate_terminal,
    select_balanced_rows,
    strict_json_loads,
    validate_bank_disjointness,
    validate_ipc1_plan,
    validate_ordered_records,
    verify_published_terminal,
    verify_seed_derivations,
)


def _repository() -> Path:
    return Path(__file__).resolve().parents[2]


def _runner_module():
    path = _repository() / "scripts/latent/run_phase_b_ipc1_matched_learning_v1.py"
    spec = importlib.util.spec_from_file_location("phase_b_ipc1_test_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool(split: str) -> list[dict[str, str]]:
    rows_per_action = 24 if split == "train" else 12
    return [
        {"task_key": f"{split}-{action}-{index:03d}", "action": action, "payload": f"p{index}"}
        for action in ACTIONS
        for index in range(rows_per_action)
    ]


def _metric_rows(*, nll: float, margin: float) -> list[dict[str, object]]:
    return [
        {"task_key": f"eval-{index:02d}", "action": ACTIONS[index % 3], "nll": nll, "margin": margin}
        for index in range(24)
    ]


def test_ipc1_seed_derivations_and_selection_are_exact_and_order_independent() -> None:
    verify_seed_derivations()
    for split in SELECTIONS:
        rows = _pool(split)
        selected, selection = select_balanced_rows(rows, split=split)
        reversed_selected, reversed_selection = select_balanced_rows(list(reversed(rows)), split=split)
        assert selected == reversed_selected
        assert selection == reversed_selection
        expected = SELECTIONS[split]["rows_per_action"]
        assert [row["action"] for row in selected] == list(ACTIONS) * expected
        assert [row["position"] for row in selection["selected"]] == list(range(len(selected)))


def test_ipc1_training_schedule_is_four_exact_balanced_updates() -> None:
    selected, selection = select_balanced_rows(_pool("train"), split="train")
    assert len(selected) == 48
    updates = selection["updates"]
    assert [record["update_index"] for record in updates] == [1, 2, 3, 4]
    assert all([row["expected_action"] for row in update["rows"]] == list(ACTIONS) * 4 for update in updates)
    assert len({row["task_key"] for update in updates for row in update["rows"]}) == 48


def test_ipc1_call_cache_and_memory_schedules_cover_both_firewall_outcomes() -> None:
    keys = {
        "train": [f"train-{index}" for index in range(48)],
        "validation": [f"validation-{index}" for index in range(24)],
        "heldout": [f"heldout-{index}" for index in range(24)],
    }
    rejected = build_model_call_schedule(**{f"{name}_keys": value for name, value in keys.items()}, open_heldout=False)
    opened = build_model_call_schedule(**{f"{name}_keys": value for name, value in keys.items()}, open_heldout=True)
    assert (len(rejected), len(opened)) == (532, 796)
    assert (len(build_cache_guard_labels(rejected)), len(build_cache_guard_labels(opened))) == (1067, 1595)
    assert sum(call["backward"] for call in rejected) == sum(call["backward"] for call in opened) == 147
    assert len(build_memory_checkpoint_labels(rejected)) == len(set(build_memory_checkpoint_labels(rejected)))


def test_ipc1_failure_progress_helpers_bind_calls_to_cache_and_update_boundaries() -> None:
    runner = _runner_module()
    keys = {
        "train": [f"train-{index}" for index in range(48)],
        "validation": [f"validation-{index}" for index in range(24)],
        "heldout": [f"heldout-{index}" for index in range(24)],
    }
    schedule = build_model_call_schedule(
        **{f"{name}_keys": value for name, value in keys.items()}, open_heldout=True
    )
    expected = build_cache_guard_labels(schedule)
    assert runner._legal_failure_cache_core(expected, calls=0, schedule_length=len(schedule)) == [
        ["CACHE_GUARD_ENTRY"],
        [],
        ["CACHE_GUARD_ENTRY", "CACHE_GUARD_PRE_IPC1_C0001"],
    ]
    assert runner._legal_failure_cache_core(expected, calls=1, schedule_length=len(schedule)) == [
        expected[:3],
        expected[:2],
        expected[:4],
    ]
    calls_before_learning = next(
        index for index, record in enumerate(schedule) if record["phase"] == "learning"
    )
    assert runner._completed_training_updates(schedule, calls_before_learning + 12) == (1, True)
    assert runner._completed_training_updates(schedule, calls_before_learning + 13) == (1, False)


def test_ipc1_failure_cache_trip_rules_are_exact_and_cache_rejection_is_unbounded() -> None:
    runner = _runner_module()
    for trip_count in (0, 1):
        runner._validate_failure_dynamic_cache_trips(
            labels=[], trip_count=trip_count, cache_rejection=False
        )
    runner._validate_failure_dynamic_cache_trips(
        labels=["CACHE_GUARD_ENTRY"], trip_count=1, cache_rejection=False
    )
    runner._validate_failure_dynamic_cache_trips(
        labels=["CACHE_GUARD_ENTRY"], trip_count=1_000_000, cache_rejection=True
    )
    for labels, trip_count, cache_rejection in (
        ([], 2, True),
        (["CACHE_GUARD_ENTRY"], 0, True),
        (["CACHE_GUARD_ENTRY"], 2, False),
    ):
        with pytest.raises(PhaseBContractError, match="DynamicCache"):
            runner._validate_failure_dynamic_cache_trips(
                labels=labels, trip_count=trip_count, cache_rejection=cache_rejection
            )


def test_ipc1_output_scale_evidence_is_full_finite_and_opens_after_step1() -> None:
    runner = _runner_module()
    runner._validate_sidecar_output_scale([], arm="STATIC", update_index=1)
    for arm in ("FFN", "RECURRENT"):
        values = [0.0] * 255 + [0.01]
        runner._validate_sidecar_output_scale(values, arm=arm, update_index=1)
        runner._validate_sidecar_output_scale([0.0] * 256, arm=arm, update_index=2)
        with pytest.raises(PhaseBContractError, match="output-scale evidence"):
            runner._validate_sidecar_output_scale(values[:-1], arm=arm, update_index=1)
        nonfinite = list(values)
        nonfinite[17] = float("nan")
        with pytest.raises(PhaseBContractError, match="finite"):
            runner._validate_sidecar_output_scale(nonfinite, arm=arm, update_index=1)
        with pytest.raises(PhaseBContractError, match="did not open"):
            runner._validate_sidecar_output_scale([0.0] * 256, arm=arm, update_index=1)


def test_ipc1_render_proof_rebinds_nonascii_source_to_bank_canonical_hash_only() -> None:
    runner = _runner_module()
    row = {
        "task_key": "document_adaptive_d0-v0-i99999:solve-anchor-1",
        "action": "solve_owned",
        "messages": [{"role": "user", "content": "identity carrier — exact"}],
    }
    bank_hash = runner.canonical_bank_sha256(row)
    inherited_hash = canonical_json_sha256(row)
    assert inherited_hash != bank_hash
    proof = {
        "task_key": row["task_key"],
        "action": row["action"],
        "source_row_sha256": inherited_hash,
        "sentinel": {"preserved": [1, 2, 3]},
    }
    selection = {
        "selected": [{"task_key": row["task_key"], "expected_action": row["action"]}],
        "row_canonical_sha256": [bank_hash],
    }
    rebound, evidence = runner._bind_render_proof_source_rows(
        [row], [proof], selection=selection, split="train"
    )
    assert proof["source_row_sha256"] == inherited_hash
    assert rebound == [{**proof, "source_row_sha256": bank_hash}]
    assert evidence == {
        "name": "train",
        "row_count": 1,
        "authoritative_bank_hash_matches": 1,
        "inherited_utf8_hash_matches": 1,
        "inherited_vs_bank_mismatches": 1,
        "overwrite_only_matches": 1,
        "post_repair_validator_matches": 0,
    }
    tampered = deepcopy(selection)
    tampered["row_canonical_sha256"][0] = "0" * 64
    with pytest.raises(PhaseBContractError, match="source binding differs"):
        runner._bind_render_proof_source_rows(
            [row], [proof], selection=tampered, split="train"
        )
    wrong_inherited = deepcopy(proof)
    wrong_inherited["source_row_sha256"] = bank_hash
    with pytest.raises(PhaseBContractError, match="source binding differs"):
        runner._bind_render_proof_source_rows(
            [row], [wrong_inherited], selection=selection, split="train"
        )
    terminal_proof = {
        "task_key": row["task_key"],
        "action": row["action"],
        "source_row_sha256": bank_hash,
        "reasoning_content_sha256": "1" * 64,
        "modified_path": "messages.2.tool_calls.0.function.arguments",
        "plain_ids_sha256": "2" * 64,
        "opening_ids_sha256": "3" * 64,
        "full_ids_sha256": "4" * 64,
        "plain_tokens": 10,
        "opening_tokens": 11,
        "full_tokens": 12,
        "counterfactual_target_sha256": {action: action * 2 for action in ACTIONS},
        "action_trie_sha256": "5" * 64,
        "action_trie_branch_count": 1,
    }
    runner._validate_render_proof_split("train", [terminal_proof], selection)
    terminal_proof["source_row_sha256"] = inherited_hash
    with pytest.raises(PhaseBContractError, match="render proof differs"):
        runner._validate_render_proof_split("train", [terminal_proof], selection)


def test_ipc1_candidate_ready_and_write_failure_boundaries_are_exact() -> None:
    runner = _runner_module()
    schedule = build_model_call_schedule(
        train_keys=[f"train-{index}" for index in range(48)],
        validation_keys=[f"validation-{index}" for index in range(24)],
        heldout_keys=[f"heldout-{index}" for index in range(24)],
        open_heldout=True,
    )
    cache_labels = build_cache_guard_labels(schedule)
    memory_labels = build_memory_checkpoint_labels(schedule)
    progress = {
        "model_calls_completed": len(schedule),
        "backward_calls_completed": 147,
        "optimizer_steps_completed": 12,
    }
    partial = {
        "cache_guard": {"labels": cache_labels[:3]},
        "execution_progress": {
            "model_calls_completed": 1,
            "backward_calls_completed": 0,
            "optimizer_steps_completed": 0,
        },
        "cuda_memory": {"ledger": []},
    }
    for record in (schedule[0], schedule[1]):
        runner._validate_failure_stage_evidence(
            partial,
            {
                "stage": record["phase"],
                "task_key": record["task_key"],
                "arm": record["arm"],
                "call_index": record["call_index"],
            },
            schedule=schedule,
            candidate_files_present=[],
        )
    with pytest.raises(PhaseBContractError, match="breadcrumb"):
        runner._validate_failure_stage_evidence(
            partial,
            {"stage": "learning", "task_key": "wrong", "arm": "STATIC", "call_index": 3},
            schedule=schedule,
            candidate_files_present=[],
        )
    ready_stop = memory_labels.index("before_candidate_writes") + 1
    ready = {
        "cache_guard": {"labels": cache_labels},
        "execution_progress": progress,
        "cuda_memory": {"ledger": [{"checkpoint": label} for label in memory_labels[:ready_stop]]},
    }
    runner._validate_failure_stage_evidence(
        ready,
        {"stage": "candidate_ready", "task_key": None, "arm": None, "call_index": len(schedule)},
        schedule=schedule,
        candidate_files_present=[],
    )
    with pytest.raises(PhaseBContractError, match="candidate-ready"):
        runner._validate_failure_stage_evidence(
            ready,
            {"stage": "candidate_ready", "task_key": None, "arm": None, "call_index": len(schedule)},
            schedule=schedule,
            candidate_files_present=["STATIC.final.pt"],
        )

    after_static = memory_labels.index("candidate:STATIC:after_write") + 1
    writing = {
        "cache_guard": {"labels": cache_labels},
        "execution_progress": progress,
        "cuda_memory": {"ledger": [{"checkpoint": label} for label in memory_labels[:after_static]]},
    }
    runner._validate_failure_stage_evidence(
        writing,
        {"stage": "candidate_write", "task_key": None, "arm": "STATIC", "call_index": len(schedule)},
        schedule=schedule,
        candidate_files_present=["STATIC.final.pt"],
    )
    missing_after_write = deepcopy(writing)
    missing_after_write["cuda_memory"]["ledger"].pop()
    with pytest.raises(PhaseBContractError, match="candidate-write memory"):
        runner._validate_failure_stage_evidence(
            missing_after_write,
            {"stage": "candidate_write", "task_key": None, "arm": "STATIC", "call_index": len(schedule)},
            schedule=schedule,
            candidate_files_present=["STATIC.final.pt"],
        )


def test_ipc1_overlap_closure_rejects_key_or_row_hash_reuse() -> None:
    selected = {
        split: select_balanced_rows(_pool(split), split=split)[0] for split in ("train", "validation", "heldout")
    }
    evidence = validate_bank_disjointness(
        selected,
        excluded_key_sets={"prior": {"prior-only"}},
        excluded_row_hash_sets={"prior": {"0" * 64}},
    )
    assert evidence["all_zero"] is True
    reused_key = selected["train"][0]["task_key"]
    with pytest.raises(PhaseBContractError, match="overlap frozen source"):
        validate_bank_disjointness(
            selected,
            excluded_key_sets={"prior": {reused_key}},
            excluded_row_hash_sets={"prior": set()},
        )


def test_ipc1_common_arm_gates_preserve_frozen_thresholds() -> None:
    actions = [ACTIONS[index % 3] for index in range(24)]
    base = _metric_rows(nll=1.0, margin=0.0)
    pre = _metric_rows(nll=1.0, margin=-0.10)
    post = _metric_rows(nll=0.99, margin=0.01)
    result = evaluate_common_arm(
        action_order=actions,
        base=base,
        pre=pre,
        post=post,
        safety_and_noncollapse=True,
    )
    assert result["passed"] is True
    failed = deepcopy(post)
    for row in failed:
        row["margin"] = -0.60
    assert (
        evaluate_common_arm(
            action_order=actions,
            base=base,
            pre=pre,
            post=failed,
            safety_and_noncollapse=True,
        )["passed"]
        is False
    )


def test_ipc1_recurrent_value_is_separate_from_common_learning() -> None:
    actions = [ACTIONS[index % 3] for index in range(24)]
    ffn = _metric_rows(nll=1.0, margin=0.0)
    recurrent = {
        f"T{depth}": _metric_rows(
            nll={1: 0.991, 2: 0.986, 4: 0.98, 8: 0.981}[depth],
            margin={1: 0.01, 2: 0.02, 4: 0.04, 8: 0.035}[depth],
        )
        for depth in EVALUATION_DEPTHS
    }
    result = evaluate_recurrent_value(
        actions=actions,
        recurrent=recurrent,
        ffn=ffn,
        retention_and_stability_passed=True,
    )
    assert result["passed"] is True
    assert result["gates"][0] == {"name": "positive_recurrence_over_ffn", "passed": True}
    result = evaluate_recurrent_value(
        actions=actions,
        recurrent=recurrent,
        ffn=ffn,
        retention_and_stability_passed=False,
    )
    assert result["passed"] is False


def test_ipc1_terminal_roundtrip_validates_parsed_bytes_and_detects_tamper(tmp_path: Path) -> None:
    def validator(receipt, *, expected_names):
        unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if receipt["receipt_sha256"] != hashlib.sha256(canonical_terminal_bytes(unhashed)).hexdigest():
            raise PhaseBContractError("internal hash differs")
        validate_ordered_records(receipt["arms"], expected_names, label="arms")

    receipt = {
        "schema_version": "test",
        "arms": [{"name": "STATIC", "value": 1}, {"name": "FFN", "value": 2}],
    }
    parsed, payload, file_hash = roundtrip_validate_terminal(
        receipt,
        validator=validator,
        validator_kwargs={"expected_names": ["STATIC", "FFN"]},
    )
    path = tmp_path / "SUCCESS.json"
    path.write_bytes(payload)
    assert (
        verify_published_terminal(
            path,
            payload,
            validator=validator,
            validator_kwargs={"expected_names": ["STATIC", "FFN"]},
        )
        == file_hash
    )
    assert json.loads(payload) == parsed
    tampered = payload.replace(b'"STATIC"', b'"FFN___"', 1)
    path.write_bytes(tampered)
    with pytest.raises(PhaseBContractError):
        verify_published_terminal(
            path,
            payload,
            validator=validator,
            validator_kwargs={"expected_names": ["STATIC", "FFN"]},
        )
    with pytest.raises(PhaseBContractError, match="repeats JSON key"):
        strict_json_loads(b'{"status":"x","status":"y"}')
    with pytest.raises(PhaseBContractError, match="non-finite"):
        strict_json_loads(b'{"value":NaN}')
    with pytest.raises(PhaseBContractError, match="finite canonical JSON"):
        canonical_terminal_bytes({"value": float("inf")})


def test_ipc1_frozen_bank_closes_exact_artifacts_and_action_orders() -> None:
    bank = _repository() / "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-bank-v1"
    manifest = json.loads((bank / "MANIFEST.json").read_text())
    assert manifest["rows"] == [
        {"count": 48, "split": "train"},
        {"count": 24, "split": "validation"},
        {"count": 24, "split": "heldout"},
    ]
    assert manifest["freshness"]["all_zero"] is True
    assert manifest["freshness"]["selected_rows_permanently_excluded_from_future_training"] == 96
    for name, expected in manifest["artifacts"].items():
        assert hashlib.sha256((bank / name).read_bytes()).hexdigest() == expected
    for split, expected_rows in (("train", 48), ("validation", 24), ("heldout", 24)):
        selection = json.loads((bank / f"{split}-selection.json").read_text())
        assert [row["position"] for row in selection["selected"]] == list(range(expected_rows))
        assert [row["expected_action"] for row in selection["selected"]] == list(ACTIONS) * (expected_rows // 3)


def test_ipc1_failure_taxonomy_does_not_misclassify_contract_errors_as_runtime() -> None:
    runner = _runner_module()
    assert runner._failure_class(runner.MechanismRejected("x")) == (
        "b_ipc1_mechanism_rejected",
        "scientific_mechanism_rejection",
    )
    assert runner._failure_class(runner.CacheContractViolated("x")) == (
        "b_ipc1_nocache_rejected",
        "scientific_cache_rejection",
    )
    assert runner._failure_class(PhaseBContractError("x")) == (
        "b_ipc1_incomplete",
        "contract_or_evidence_incomplete",
    )
    assert runner._failure_class(RuntimeError("CUDA out of memory")) == (
        "infrastructure_invalid",
        "infrastructure",
    )
    assert runner._failure_class(RuntimeError("ordinary model runtime defect")) == (
        "b_ipc1_incomplete",
        "contract_or_evidence_incomplete",
    )


def test_ipc1_failure_module_records_preserve_stagewise_absence_and_list_order() -> None:
    runner = _runner_module()
    runner._validate_failure_module_records(None, allow_absent=True, label="initial")
    runner._validate_failure_module_records([], allow_absent=True, label="current")
    records = [
        {"name": name, "sha256": hashlib.sha256(name.encode()).hexdigest()}
        for name in runner.MODULE_NAMES
    ]
    runner._validate_failure_module_records(records, allow_absent=False, label="current")
    mapping_reordered = [dict(reversed(list(record.items()))) for record in records]
    parsed = strict_json_loads(canonical_terminal_bytes({"records": mapping_reordered}))
    runner._validate_failure_module_records(parsed["records"], allow_absent=False, label="current")
    with pytest.raises(PhaseBContractError, match="module order differs"):
        runner._validate_failure_module_records(list(reversed(records)), allow_absent=False, label="current")
    with pytest.raises(PhaseBContractError, match="module order differs"):
        runner._validate_failure_module_records([], allow_absent=False, label="current")


def test_ipc1_plan_and_terminal_writers_reject_schema_or_global_terminal_tamper(tmp_path: Path) -> None:
    repository = _repository()
    plan = json.loads(
        (
            repository
            / "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-r1-matched-learning-run1-plan.json"
        ).read_text()
    )
    plan["terminal_proof"]["late_failure_tamper_cases_rejected"] = 21
    plan["plan_sha256"] = canonical_plan_sha256(plan)
    validate_ipc1_plan(plan)
    tampered = deepcopy(plan)
    tampered["unexpected"] = True
    with pytest.raises(PhaseBContractError, match="plan keyset differs"):
        validate_ipc1_plan(tampered)

    runner = _runner_module()
    output = tmp_path / "terminal"
    output.mkdir()
    payload = canonical_terminal_bytes({"terminal": "SUCCESS"})
    path = runner._atomic_publish_bytes(output, "SUCCESS.json", payload)
    assert path.read_bytes() == payload
    assert not list(output.glob(".*.tmp"))
    with pytest.raises(FileExistsError, match="globally fresh"):
        runner._atomic_publish_bytes(output, "FAILURE.json", canonical_terminal_bytes({"terminal": "FAILURE"}))
    assert sorted(item.name for item in output.iterdir()) == ["SUCCESS.json"]


def test_ipc1_exact_host_terminal_proof_closure_is_byte_bound() -> None:
    proof = _repository() / "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-terminal-proof-v1"
    manifest = json.loads((proof / "MANIFEST.json").read_text())
    success = manifest["successful_exact_host_proof"]
    encoded = proof / success["proof_base64_path"]
    assert hashlib.sha256(encoded.read_bytes()).hexdigest() == success["proof_base64_file_sha256"]
    encoded_bytes = encoded.read_bytes()
    assert encoded_bytes.endswith(b"\n") and not any(byte in b" \t\r\n" for byte in encoded_bytes[:-1])
    decoded = base64.b64decode(encoded_bytes[:-1], validate=True)
    assert hashlib.sha256(decoded).hexdigest() == success["decoded_proof_sha256"]
    receipt = strict_json_loads(decoded)
    assert receipt["execution_commit"] == manifest["proof_execution_commit"]
    assert receipt["runner_sha256"] == manifest["runner_sha256"]
    assert receipt["model_loaded"] is False
    assert receipt["cuda_initialized"] is False
    assert receipt["exact_host_repository"] is True
    assert len(receipt["tamper_cases_rejected"]) == success["tamper_cases_rejected"]
    for prefix in ("log", "exit_status"):
        path = proof / success[f"{prefix}_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == success[f"{prefix}_sha256"]
    failed = manifest["superseded_preexecution_command_failure"]
    for prefix in ("log", "exit_status"):
        path = proof / failed[f"{prefix}_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == failed[f"{prefix}_sha256"]
    validator_failure = manifest["superseded_validator_proof_failure"]
    assert validator_failure["proof_file_created"] is False
    assert validator_failure["failure_terminal_count"] == 0
    assert validator_failure["model_loaded"] is False
    assert validator_failure["cuda_initialized"] is False
    for prefix in ("log", "exit_status"):
        path = proof / validator_failure[f"{prefix}_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == validator_failure[f"{prefix}_sha256"]
    tamper_failure = manifest["superseded_late_tamper_proof_failure"]
    assert tamper_failure["proof_file_created"] is False
    assert tamper_failure["late_failure_terminal_count"] == 0
    assert tamper_failure["model_loaded"] is False
    assert tamper_failure["cuda_initialized"] is False
    assert [record["status"] for record in tamper_failure["ordinary_failure_files"]] == [
        pair[0] for pair in FAILURE_STATUS_CLASSES
    ]
    for prefix in ("log", "exit_status"):
        path = proof / tamper_failure[f"{prefix}_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == tamper_failure[f"{prefix}_sha256"]
