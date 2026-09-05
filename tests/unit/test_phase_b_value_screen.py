import ast
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from prime_rl.phase_b_contract import PhaseBContractError
from prime_rl.phase_b_value_screen import (
    ACTIONS,
    EVALUATION_DEPTHS,
    TRAINING_ARMS,
    action_margin_from_logits,
    build_action_trie,
    canonical_plan_sha256,
    evaluate_nomination,
    paired_loss_deltas,
    validate_evaluation_keys,
    validate_training_batches,
)


def _repository() -> Path:
    return Path(__file__).resolve().parents[2]


def _runner_module():
    path = _repository() / "scripts/latent/run_phase_b_teacher_forced_value_screen_v1.py"
    spec = importlib.util.spec_from_file_location("phase_b1r_test_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _balanced_rows() -> list[dict[str, str]]:
    selection = json.loads(
        (
            _repository() / "experiments/qwen35-2b-latent-coordinator-v1/phase-b-b1-bank-v1/training-selection.json"
        ).read_text()
    )
    return [{"task_key": key, "action": ACTIONS[index % 3]} for index, key in enumerate(selection["task_keys"])]


def _balanced_selection(rows: list[dict[str, str]]) -> dict[str, object]:
    return {"batches": [[row["task_key"] for row in rows[index : index + 12]] for index in range(0, 48, 12)]}


def test_learning_screen_arm_and_depth_constants_are_fixed() -> None:
    assert TRAINING_ARMS == ("STATIC", "FFN", "RECURRENT")
    assert EVALUATION_DEPTHS == (1, 2, 4, 8)


def test_plan_internal_hash_omits_only_its_self_referential_field() -> None:
    payload = {"schema_version": "example", "plan_sha256": "ignored", "nested": {"value": 3}}
    expected = hashlib.sha256(
        json.dumps(
            {"schema_version": "example", "nested": {"value": 3}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert canonical_plan_sha256(payload) == expected
    changed = {**payload, "nested": {"value": 4}}
    assert canonical_plan_sha256(changed) != expected
    with pytest.raises(PhaseBContractError, match="internal canonical hash"):
        canonical_plan_sha256({"schema_version": "example"})


def test_b1r_cuda_memory_cap_checks_current_and_peak_allocated_and_reserved() -> None:
    runner = _runner_module()

    class FakeCuda:
        values = {
            "memory_allocated": 10,
            "memory_reserved": 20,
            "max_memory_allocated": 30,
            "max_memory_reserved": 40,
        }

        def __getattr__(self, name):
            return lambda _device: self.values[name]

    torch = SimpleNamespace(cuda=FakeCuda())
    audit = {"cuda_memory_contract": {"cap_bytes": runner.CUDA_MEMORY_CAP_BYTES, "checkpoint_ledger": []}}
    snapshot = runner._enforce_cuda_memory_cap(torch, audit, "bounded")
    assert snapshot == {
        "checkpoint": "bounded",
        "current_allocated_bytes": 10,
        "current_reserved_bytes": 20,
        "maximum_allocated_bytes": 30,
        "maximum_reserved_bytes": 40,
    }
    torch.cuda.values["max_memory_reserved"] = runner.CUDA_MEMORY_CAP_BYTES + 1
    with pytest.raises(runner.ResourceContractExceeded, match="maximum_reserved_bytes"):
        runner._enforce_cuda_memory_cap(torch, audit, "violation")
    assert audit["cuda_memory_contract"]["checkpoint_ledger"][-1]["checkpoint"] == "violation"


def test_training_batches_are_four_disjoint_balanced_updates_covering_all_rows() -> None:
    rows = _balanced_rows()
    batches = validate_training_batches(rows, _balanced_selection(rows))

    assert len(batches) == 4
    assert all(len(batch.task_keys) == 12 for batch in batches)
    flattened = [key for batch in batches for key in batch.task_keys]
    assert len(flattened) == len(set(flattened)) == 48
    by_key = {row["task_key"]: row for row in rows}
    for batch in batches:
        assert Counter(by_key[key]["action"] for key in batch.task_keys) == Counter({action: 4 for action in ACTIONS})


def test_training_batches_reject_reuse_and_action_imbalance() -> None:
    rows = _balanced_rows()
    selection = _balanced_selection(rows)
    selection["batches"][1][0] = selection["batches"][0][0]  # type: ignore[index]
    with pytest.raises(PhaseBContractError):
        validate_training_batches(rows, selection)


def test_evaluation_keys_require_disjoint_exact_action_counts() -> None:
    selection = json.loads(
        (
            _repository() / "experiments/qwen35-2b-latent-coordinator-v1/phase-b-b1-bank-v1/heldout-selection.json"
        ).read_text()
    )
    rows = [{"task_key": item["task_key"], "action": item["expected_action"]} for item in selection["key_actions"]]
    assert len(validate_evaluation_keys({"train-only"}, rows, selection)) == 12
    with pytest.raises(PhaseBContractError, match="overlap training"):
        validate_evaluation_keys({rows[0]["task_key"]}, rows, selection)


def test_paired_loss_deltas_preserve_task_pairing_and_sign() -> None:
    metrics = {
        "RECURRENT_T4": [{"task_key": "a", "loss": 0.2}, {"task_key": "b", "loss": 0.7}],
        "FFN": [{"task_key": "a", "loss": 0.5}, {"task_key": "b", "loss": 0.6}],
    }

    assert paired_loss_deltas(metrics, "RECURRENT_T4", "FFN") == [
        {"task_key": "a", "left_minus_right": pytest.approx(-0.3)},
        {"task_key": "b", "left_minus_right": pytest.approx(0.1)},
    ]
    metrics["FFN"].append({"task_key": "c", "loss": 0.4})
    with pytest.raises(PhaseBContractError, match="task keys differ"):
        paired_loss_deltas(metrics, "RECURRENT_T4", "FFN")


def test_action_trie_uses_only_correct_path_branch_logits() -> None:
    trie = build_action_trie(
        {
            "solve_owned": [10, 20, 30],
            "delegate_terminal": [10, 21, 31],
            "delegate_coordinator": [10, 21, 32],
        },
        correct_action="delegate_terminal",
    )
    values = {(1, 21): 5.0, (1, 20): 3.0, (2, 31): 2.0, (2, 32): 2.5}
    margin, branches = action_margin_from_logits(trie, lambda offset, token: values[offset, token])

    assert [branch["target_offset"] for branch in branches] == [1, 2]
    assert margin == pytest.approx(-0.5)
    assert trie["branch_count"] == 2
    with pytest.raises(PhaseBContractError, match="prefixes"):
        build_action_trie(
            {"solve_owned": [1], "delegate_terminal": [1, 2], "delegate_coordinator": [3]},
            correct_action="solve_owned",
        )


def _nomination_metrics() -> tuple[dict[str, list[dict[str, object]]], dict[str, list[dict[str, object]]]]:
    names = ("STATIC", "FFN", "RECURRENT_T1", "RECURRENT_T2", "RECURRENT_T4", "RECURRENT_T8")
    initial = {name: [] for name in names}
    final = {name: [] for name in names}
    post_nll = {
        "STATIC": 0.99,
        "FFN": 0.99,
        "RECURRENT_T1": 0.985,
        "RECURRENT_T2": 0.982,
        "RECURRENT_T4": 0.98,
        "RECURRENT_T8": 0.981,
    }
    post_margin = {
        "STATIC": 0.01,
        "FFN": 0.01,
        "RECURRENT_T1": 0.02,
        "RECURRENT_T2": 0.03,
        "RECURRENT_T4": 0.04,
        "RECURRENT_T8": 0.035,
    }
    for index in range(12):
        common = {"task_key": f"heldout-{index}", "action": ACTIONS[index % 3]}
        for name in names:
            initial[name].append({**common, "nll": 1.0, "margin": 0.0})
            row: dict[str, object] = {**common, "nll": post_nll[name], "margin": post_margin[name]}
            if name == "RECURRENT_T8":
                row["retention"] = {
                    f"T{depth}": {"cosine": 0.999, "norm_ratio": 1.0, "relative_l2": 0.01}
                    for depth in EVALUATION_DEPTHS
                }
                row["stability_T8"] = {
                    "memory_change_rms": [1.0, 0.8, 0.64, 0.512, 0.41, 0.328, 0.262, 0.21],
                    "median_memory_contraction_steps_2_8": 0.8,
                    "max_memory_contraction_steps_2_8": 0.8,
                    "memory_oscillation_rate": 0.0,
                    "finite": True,
                }
            final[name].append(row)
    return initial, final


def test_nomination_gates_are_exact_and_scientific_miss_is_valid() -> None:
    initial, final = _nomination_metrics()
    result = evaluate_nomination(initial, final, safety_gate_passed=True)
    assert result["nominated"] is True
    assert result["disposition"] == "b1_nominated"
    result = evaluate_nomination(initial, final, safety_gate_passed=False)
    assert result["nominated"] is False
    assert result["disposition"] == "b1_not_nominated"


def test_runner_has_exact_update_surface_and_no_promotion_or_optimizer_checkpoint() -> None:
    path = _repository() / "scripts/latent/run_phase_b_teacher_forced_value_screen_v1.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert "for arm in TRAINING_ARMS" in source
    assert "for batch in batches" in source
    assert "output.loss / len(batch.task_keys)" in source
    assert '"optimizer_state_persisted": False' in source
    assert "optimizers: dict" not in source
    assert "optimizer.state.clear()" in source
    assert "evaluate_nomination(" in source
    assert source.index("nomination = evaluate_nomination(") < source.index("checkpoint_payloads = {")
    assert source.index("immutable_input_hashes = {") < source.index("checkpoint_payloads = {")
    assert '"minimum_complete_live_trajectories_unchanged": 4' in source
    assert '"teacher_forced_rows_count_as_live_trajectories": False' in source
    assert '"valid_only_with_exact_terminal_receipt": "SUCCESS.json"' in source
    assert source.count('"invalid_unclassified_do_not_use"') == 2
    assert '"present_file_hashes": post_failure_hash_audit.get("compact_checkpoint_hashes", {})' in source
    assert "torch.cuda.set_per_process_memory_fraction(requested_fraction, 0)" in source
    assert "torch.cuda.reset_peak_memory_stats(0)" in source
    assert 'f"checkpoint:{name}:before_write"' in source
    assert 'f"checkpoint:{name}:after_write"' in source
    assert '"before_success"' in source
    assert ".generate(" not in source
    assert "use_cache=True" not in source
    assert not any(
        isinstance(call.func, ast.Attribute) and call.func.attr in {"save_pretrained", "push_to_hub"} for call in calls
    )
    saved_names = {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"STATIC.final.pt", "FFN.final.pt", "RECURRENT.final.pt"}.issubset(saved_names)


def test_frozen_bank_has_exact_portable_provenance_closure() -> None:
    bank = _repository() / "experiments/qwen35-2b-latent-coordinator-v1/phase-b-b1-bank-v1"
    manifest = json.loads((bank / "MANIFEST.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert hashlib.sha256((bank / name).read_bytes()).hexdigest() == expected
    assert manifest["taskset"]["commit"] == "5283a85a01b5e8a065b3d2db17f9efa6aa0f3b2f"
    assert manifest["taskset"]["source_sha256"] == ("15332134b09d5b6bccfaaf03166ddb5aa88b8bafe84a01e34b7409b218f50499")
    assert manifest["row_list_canonical_sha256"] == ("94de16b26d8b4fbc99b22ab0e1312933e3a83e5f79a521d2a6327f2f28fae988")
    assert manifest["freshness"]["exact_task_key_overlap_with_e33_and_b1_training"] == 0
    assert "previously evaluated" in manifest["freshness"]["disclosure"]


def test_tokenizer_preflight_precedes_torch_and_model_imports() -> None:
    source = (_repository() / "scripts/latent/run_phase_b_teacher_forced_value_screen_v1.py").read_text(
        encoding="utf-8"
    )

    tokenizer_call = source.index("context = tokenizer_preflight(")
    torch_import = source.index("        import torch", tokenizer_call)
    model_import = source.index("from transformers import AutoModelForImageTextToText", tokenizer_call)
    output_creation = source.index("args.output_dir.mkdir", tokenizer_call)
    assert tokenizer_call < output_creation < torch_import < model_import
