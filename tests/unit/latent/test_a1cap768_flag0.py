import ast
import copy
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import prime_rl.latent.a1cap768_flag0 as flag0
from prime_rl.latent.a0 import canonical_json_hash


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_frozen_flag0_constants_and_run4_evidence_are_exact():
    flag0._validate_constants()
    assert canonical_json_hash(flag0.FIXTURE) == flag0.FIXTURE_SHA256
    assert canonical_json_hash(flag0.OPERATION_SCHEDULE) == flag0.OPERATION_SCHEDULE_SHA256
    assert canonical_json_hash(flag0.FLAG_NAMES) == flag0.FLAG_NAMES_SHA256
    assert canonical_json_hash(flag0.RUN4_FLAG_NAMES) == flag0.RUN4_FLAG_NAMES_SHA256
    assert canonical_json_hash(flag0.COMPARISON_SCHEDULE) == flag0.COMPARISON_SCHEDULE_SHA256
    assert canonical_json_hash(flag0.memory_labels()) == flag0.MEMORY_LABELS_SHA256
    assert len(flag0.OPERATION_SCHEDULE) == 7
    assert len(flag0.FLAG_NAMES) == 25
    assert len(flag0.RUN4_FLAG_NAMES) == 16
    assert len(flag0.COMPARISON_SCHEDULE) == 13
    assert len(flag0.memory_labels()) == len(set(flag0.memory_labels())) == 17
    assert flag0.validate_run4_rejection(Path(".")) == flag0.RUN4_REJECTION_EVIDENCE


def test_run4_evidence_fails_closed_on_byte_tamper(tmp_path):
    relative = Path("experiments/qwen35-2b-latent-workspace-v1")
    target = tmp_path / relative
    target.mkdir(parents=True)
    for name in (
        "a1-nc0-cap768-plan-v1.json",
        "a1-nc0-cap768-run4-mechanism-rejection-failure.json",
        "a1-nc0-cap768-run4-mechanism-rejection-run.log",
    ):
        shutil.copy2(relative / name, target / name)
    assert flag0.validate_run4_rejection(tmp_path) == flag0.RUN4_REJECTION_EVIDENCE
    with (target / "a1-nc0-cap768-run4-mechanism-rejection-run.log").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="run4 rejection evidence changed"):
        flag0.validate_run4_rejection(tmp_path)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (flag0.NoCacheRejected("x"), "capture768_nocache_rejected"),
        (flag0.ResourceFitRejected("x"), "capture768_resource_fit_rejected"),
        (flag0.DiagnosticIncomplete("x"), "capture768_flag_isolation_incomplete"),
        (TimeoutError("x"), "infrastructure_invalid"),
        (ValueError("x"), "infrastructure_invalid"),
    ],
)
def test_flag0_terminal_taxonomy(error, status):
    assert flag0.classify_failure(error)[0] == status


def test_flag0_launcher_freezes_namespace_resources_and_shared_environment():
    shell = Path("scripts/latent/run_a1_nc0_cap768_flag0_v1.sh").read_text()
    declarations = shell.split("\nfor asset in ", 1)[0]
    valid = subprocess.run(
        [
            "bash",
            "-c",
            declarations + '\nprintf "%s\\n" "$shared_venv"',
            "launcher",
            "a" * 40,
            "b" * 64,
            flag0.AUTHORIZED_RUN_ID,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0
    assert valid.stdout == "/home/ubuntu/rlm/prime-rl/.venv\n"
    assert "62914560" in shell and "67108864" in shell
    assert "CUDA_VISIBLE_DEVICES=0" in shell
    assert "3600s" in shell and "134217728" in shell
    reused = subprocess.run(
        ["bash", "-c", declarations, "launcher", "a" * 40, "b" * 64, "a1-nc0-cap768-run4"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert reused.returncode == 64


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _flags_assignment(function: ast.FunctionDef) -> ast.Assign:
    return next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "flags" for target in node.targets)
    )


def test_flag0_scientific_core_preserves_cap_flag_expressions_and_operation_order():
    cap_tree = ast.parse(Path("scripts/latent/run_a1_nc0_cap768_v1.py").read_text())
    flag_tree = ast.parse(Path("scripts/latent/run_a1_nc0_cap768_flag0_v1.py").read_text())
    cap_flags = _flags_assignment(_function(cap_tree, "_case")).value
    diagnostic = _function(flag_tree, "_run_case")
    diagnostic_flags = _flags_assignment(diagnostic).value
    assert isinstance(cap_flags, ast.Dict) and isinstance(diagnostic_flags, ast.Dict)
    assert [key.value for key in cap_flags.keys] == flag0.RUN4_FLAG_NAMES
    assert [key.value for key in diagnostic_flags.keys] == flag0.RUN4_FLAG_NAMES
    assert [ast.dump(value) for value in cap_flags.values] == [ast.dump(value) for value in diagnostic_flags.values]
    source = ast.unparse(diagnostic)
    order = [
        source.index("model.get_input_embeddings()(padded)"),
        source.index("id1, id1_timing = _forward"),
        source.index("e1, e1_timing = _forward"),
        source.index("e2, e2_timing = _forward"),
        source.index("id0, id0_timing = _forward"),
        source.index("model.lm_head(hidden)"),
    ]
    assert order == sorted(order)
    assert "torch.inference_mode()" in source
    assert "not all(flags.values())" not in source


def test_comparison_records_exact_bitwise_and_nonfinite_evidence():
    equal = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    same = equal.clone()
    row = flag0.COMPARISON_SCHEDULE[0]
    import importlib.util

    path = Path("scripts/latent/run_a1_nc0_cap768_flag0_v1.py")
    spec = importlib.util.spec_from_file_location("flag0_runner_for_test", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    observed = runner._comparison(row, equal, same)
    assert observed["torch_equal"] is True
    assert observed["mismatch_count"] == observed["count_nonzero"] == 0
    assert observed["first_flat_mismatch"] is None
    assert observed["metrics_defined"] is True
    assert observed["max_abs"] == observed["rms_diff"] == observed["normalized_rms"] == 0.0
    nonfinite = runner._comparison(row, torch.tensor([float("nan")]), torch.tensor([float("nan")]))
    assert nonfinite["metrics_defined"] is False
    assert all(nonfinite[key] is None for key in ("max_abs", "rms_diff", "rhs_rms", "normalized_rms"))


def _valid_plan() -> dict[str, object]:
    return {
        "plan_sha256": digest("plan"),
        "mechanism_code_commit": "a" * 40,
        "asset_sha256": {"scripts/latent/run_a1_nc0_cap768_flag0_v1.py": digest("runner")},
        "protected_checkpoints": {
            "coordinator_e33": flag0._E33,
            "worker_h176": flag0._H176,
        },
    }


def _tensor_evidence() -> dict[str, dict[str, object]]:
    shapes = {
        "exact_embeddings": [1, 768, 2048],
        "L_ID_KEEP1.logits": [1, 1, 248320],
        "L_ID_KEEP1.hidden": [1, 768, 2048],
        "L_ID_KEEP1.capture": [1, 128, 2048],
        "L_E_KEEP1.logits": [1, 1, 248320],
        "L_E_KEEP1.hidden": [1, 768, 2048],
        "L_E_KEEP1.capture": [1, 128, 2048],
        "L_E_REPEAT_KEEP1.logits": [1, 1, 248320],
        "L_E_REPEAT_KEEP1.hidden": [1, 768, 2048],
        "L_E_REPEAT_KEEP1.capture": [1, 128, 2048],
        "L_ID_KEEP0_CONTROL.logits": [1, 768, 248320],
        "L_ID_KEEP0_CONTROL.last_logits": [1, 1, 248320],
        "L_ID_KEEP0_CONTROL.hidden": [1, 768, 2048],
        "L_ID_KEEP0_CONTROL.capture": [1, 128, 2048],
        "PROJ_ID1_LAST.logits": [1, 1, 248320],
        "PROJ_ID0_LAST.logits": [1, 1, 248320],
    }
    classes = {
        "exact_embeddings": "embed",
        "L_ID_KEEP1.logits": "head",
        "L_E_KEEP1.logits": "head",
        "L_E_REPEAT_KEEP1.logits": "head",
        "PROJ_ID1_LAST.logits": "head",
        "PROJ_ID0_LAST.logits": "head",
        "L_ID_KEEP0_CONTROL.last_logits": "full-head",
        "L_ID_KEEP0_CONTROL.logits": "full-logits",
    }
    for prefix in ("L_ID_KEEP1", "L_E_KEEP1", "L_E_REPEAT_KEEP1", "L_ID_KEEP0_CONTROL"):
        classes[f"{prefix}.hidden"] = "hidden"
        classes[f"{prefix}.capture"] = "capture"
    return {
        name: {"dtype": "torch.bfloat16", "shape": shape, "sha256": digest(classes[name])}
        for name, shape in shapes.items()
    }


def _valid_receipt(plan: dict[str, object]) -> dict[str, object]:
    tensors = _tensor_evidence()
    flags = dict.fromkeys(flag0.FLAG_NAMES, True)
    flags["keep0_last_logits_keep1_bitwise"] = False
    flags["proj_id0_matches_id0_last_logits_bitwise"] = False
    name_to_flag = {
        spec["name"]: next(name for name in flag0.FLAG_NAMES if name == f"{spec['name']}_bitwise")
        for spec in flag0.COMPARISON_SCHEDULE
    }
    comparisons = []
    for spec in flag0.COMPARISON_SCHEDULE:
        equal = flags[name_to_flag[spec["name"]]]
        rhs_rms = 1.0
        rms = 0.0 if equal else 1.0
        comparisons.append(
            {
                **spec,
                "lhs_dtype": "torch.bfloat16",
                "rhs_dtype": "torch.bfloat16",
                "lhs_shape": tensors[spec["lhs"]]["shape"],
                "rhs_shape": tensors[spec["rhs"]]["shape"],
                "lhs_sha256": tensors[spec["lhs"]]["sha256"],
                "rhs_sha256": tensors[spec["rhs"]]["sha256"],
                "torch_equal": equal,
                "element_count": 248320,
                "mismatch_count": 0 if equal else 1,
                "count_nonzero": 0 if equal else 1,
                "first_flat_mismatch": None if equal else 0,
                "metrics_defined": True,
                "max_abs": rms,
                "rms_diff": rms,
                "rhs_rms": rhs_rms,
                "normalized_rms": rms,
            }
        )
    operations = [{**spec, "cuda_event_seconds": 1.0, "wall_seconds": 1.0} for spec in flag0.OPERATION_SCHEDULE]
    ledger = [
        {
            "label": label,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
        for label in flag0.memory_labels()
    ]
    source = {
        name: {"path": f"/frozen/{name}.py", "sha256": sha}
        for name, sha in flag0._RUNTIME["transformers_source_sha256"].items()
    }
    physical = {
        "names": [flag0.RESOURCE_BOUNDS["gpu_model"]] * 2,
        "uuids": ["GPU-0", "GPU-1"],
        "memory_used_mib": [1, 1],
        "compute_apps": [],
    }
    receipt = {
        "schema_version": flag0.RECEIPT_SCHEMA,
        "status": "capture768_flag_isolation_complete",
        "plan_sha256": plan["plan_sha256"],
        "mechanism_code_commit": plan["mechanism_code_commit"],
        "execution_commit": "b" * 40,
        "asset_sha256": plan["asset_sha256"],
        "run_id": flag0.AUTHORIZED_RUN_ID,
        "fixture": flag0.FIXTURE,
        "fixture_sha256": flag0.FIXTURE_SHA256,
        "train_bank_sha256": flag0.TRAIN_BANK_SHA256,
        "operation_schedule": flag0.OPERATION_SCHEDULE,
        "operation_schedule_sha256": flag0.OPERATION_SCHEDULE_SHA256,
        "operation_counts": {
            "embedding_lookup": 1,
            "e33_forward": 4,
            "lm_head_projection": 2,
            "capture": 4,
            "generation": 0,
            "h176_forward": 0,
            "bridge": 0,
            "optimizer": 0,
            "backward": 0,
            "step": 0,
            "checkpoint": 0,
            "candidate": 0,
        },
        "flag_names": flag0.FLAG_NAMES,
        "flag_names_sha256": flag0.FLAG_NAMES_SHA256,
        "run4_flag_names": flag0.RUN4_FLAG_NAMES,
        "run4_flag_names_sha256": flag0.RUN4_FLAG_NAMES_SHA256,
        "flags": flags,
        "run4_aggregate_reproduced": True,
        "comparison_schedule": flag0.COMPARISON_SCHEDULE,
        "comparison_schedule_sha256": flag0.COMPARISON_SCHEDULE_SHA256,
        "comparisons": comparisons,
        "tensor_evidence": tensors,
        "input_evidence": {
            "rendered_ids_shape": [1, 517],
            "rendered_ids_dtype": "torch.int64",
            "rendered_ids_contiguous": True,
            **{
                key: digest(key)
                for key in (
                    "rendered_ids_sha256",
                    "padded_ids_sha256",
                    "attention_mask_sha256",
                    "position_ids_sha256",
                    "capture_mask_sha256",
                )
            },
        },
        "run4_rejection_evidence": flag0.RUN4_REJECTION_EVIDENCE,
        "versions": {
            key: flag0._RUNTIME[key]
            for key in ("python", "transformers", "flash_linear_attention", "torch_distribution", "torch_runtime")
        },
        "runtime_sources": source,
        "static_guard": {"runner_sha256": digest("runner"), "forbidden_calls": []},
        "protected_hashes_before": plan["protected_checkpoints"],
        "protected_hashes_after": plan["protected_checkpoints"],
        "checkpoint_metadata_before": {
            "coordinator_e33": flag0._RUNTIME["checkpoint_metadata_sha256"],
            "worker_h176": flag0._RUNTIME["checkpoint_metadata_sha256"],
        },
        "checkpoint_metadata_after": {
            "coordinator_e33": flag0._RUNTIME["checkpoint_metadata_sha256"],
            "worker_h176": flag0._RUNTIME["checkpoint_metadata_sha256"],
        },
        "e33_state_tree_before": digest("state"),
        "e33_state_tree_after": digest("state"),
        "e33_parameters_frozen_no_grad": True,
        "worker_h176_loaded": False,
        "model_runtime": {
            "class": flag0._RUNTIME["model_class"],
            "hidden_size": 2048,
            "vocab_size": 248320,
            "dtype": "torch.bfloat16",
            "device": "cuda:0",
        },
        "no_cache_contract": {
            "calls": 4,
            "use_cache_false": True,
            "pkv_input_none": True,
            "pkv_output_none": True,
            "rope_reset_every_call": True,
            "model_config_use_cache": False,
            "generation_config_use_cache": False,
        },
        "cache_guard": {
            "classes": flag0._CACHE_CLASS_CLOSURE,
            "negative_control_dynamic_cache_tripped": True,
            "closure_check_count": 11,
            "restored_in_finally": True,
        },
        "memory_ledger": ledger,
        "memory_labels_sha256": flag0.MEMORY_LABELS_SHA256,
        "resources": {
            "gpu_name": flag0.RESOURCE_BOUNDS["gpu_model"],
            "total_gpu_memory_bytes": 47 * 2**30,
            "allocator_cap_bytes": 40 * 2**30,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "host_ram_bytes": 64 * 2**30,
            "free_disk_bytes_preflight": 60 * 2**30,
            "cuda_visible_devices": "0",
            "network_disabled": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            "physical_gpu_before": physical,
            "physical_gpu_after": physical,
            "physical_gpu1_unused_before_after": True,
        },
        "timings": {
            "operations": operations,
            "operation_cuda_event_seconds_sum": 7.0,
            "operation_wall_seconds_sum": 7.0,
            "tokenizer_load_seconds": 1.0,
            "model_load_seconds": 1.0,
            "compute_seconds": 10.0,
            "audit_seconds": 1.0,
            "total_seconds": 11.0,
        },
        "decision_boundary": {
            **flag0.DECISION_BOUNDARY,
            "causal_interpretation": "bf16_lm_head_shape_rounding_redesign_discussion_only",
        },
        "interpretation_boundary": flag0.INTERPRETATION,
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = canonical_json_hash(receipt, omitted_fields=("receipt_sha256",))
    return receipt


def test_receipt_validator_accepts_complete_false_flag_evidence_and_rejects_tamper():
    plan = _valid_plan()
    receipt = _valid_receipt(plan)
    flag0.validate_receipt(receipt, plan=plan)
    for path, value in (
        (("flags", "keep0_last_logits_keep1_bitwise"), True),
        (("memory_ledger", 2, "label"), "wrong"),
        (("comparisons", 0, "lhs_sha256"), digest("wrong")),
        (("decision_boundary", "training_authorized"), True),
    ):
        changed = copy.deepcopy(receipt)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        changed["receipt_sha256"] = canonical_json_hash(changed, omitted_fields=("receipt_sha256",))
        with pytest.raises(flag0.DiagnosticIncomplete):
            flag0.validate_receipt(changed, plan=plan)
