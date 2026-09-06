import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from prime_rl.phase_b_contract import PhaseBContractError
from prime_rl.phase_b_ipc1 import (
    ACTIONS,
    EVALUATION_DEPTHS,
    SELECTIONS,
    build_cache_guard_labels,
    build_memory_checkpoint_labels,
    build_model_call_schedule,
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


def test_ipc1_plan_and_terminal_writers_reject_schema_or_global_terminal_tamper(tmp_path: Path) -> None:
    repository = _repository()
    plan = json.loads(
        (
            repository / "experiments/qwen35-2b-latent-coordinator-v1/phase-b-ipc1-matched-learning-run1-plan.json"
        ).read_text()
    )
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
