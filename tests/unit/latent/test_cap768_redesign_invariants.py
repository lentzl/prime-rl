from pathlib import Path

import pytest

from prime_rl.latent.cap768_redesign_invariants import (
    InvariantViolation,
    inspect_no_training_runner,
    require_pre_model_static_guard,
    validate_comparison_partition,
    validate_exact_failure_binding,
)


def test_static_guard_ignores_its_own_string_literals_and_scans_real_flag_runner():
    path = Path("scripts/latent/run_a1_nc0_cap768_flag0_v1.py")
    evidence = inspect_no_training_runner(path)
    assert evidence.forbidden_calls == ()
    assert evidence.forbidden_identifiers == ()
    assert evidence.forbidden_imports == ()
    assert len(evidence.runner_sha256) == 64


@pytest.mark.parametrize(
    "source",
    [
        "model.generate()",
        "loss.backward()",
        "optimizer.step()",
        "from package import WorkspaceBridge",
        "optimizer = torch.optim.AdamW([])",
    ],
)
def test_static_guard_rejects_ast_training_and_generation_syntax(tmp_path, source):
    path = tmp_path / "runner.py"
    path.write_text(source)
    with pytest.raises(InvariantViolation, match="forbidden training/generation syntax"):
        inspect_no_training_runner(path)


def test_static_guard_does_not_match_constants_or_comments(tmp_path):
    path = tmp_path / "runner.py"
    path.write_text(
        "# WorkspaceBridge, AdamW, .generate(), .backward(), .step()\n"
        'MARKERS = ("WorkspaceBridge", "AdamW", "generate", "backward", "step")\n'
    )
    assert inspect_no_training_runner(path).forbidden_calls == ()


def test_pre_model_guard_order_requires_one_early_call(tmp_path):
    path = tmp_path / "runner.py"
    path.write_text(
        "def run():\n"
        "    evidence = static_guard()\n"
        "    model = Loader.from_pretrained('frozen')\n"
        "    return evidence, model\n"
    )
    require_pre_model_static_guard(path, run_function="run", guard_function="static_guard")
    path.write_text("def run():\n    model = Loader.from_pretrained('frozen')\n    return static_guard(), model\n")
    with pytest.raises(InvariantViolation, match="before model loading"):
        require_pre_model_static_guard(path, run_function="run", guard_function="static_guard")


def test_comparison_partition_keeps_full_matrix_control_descriptive():
    rows = [
        {
            "name": "hidden",
            "role": "gate",
            "lhs_shape": [1, 768, 2048],
            "rhs_shape": [1, 768, 2048],
            "torch_equal": True,
        },
        {
            "name": "keep0_full_matrix",
            "role": "descriptive_only",
            "lhs_shape": [1, 1, 248320],
            "rhs_shape": [1, 768, 248320],
            "torch_equal": False,
        },
    ]
    validate_comparison_partition(
        rows,
        gated_names=("hidden",),
        descriptive_names=("keep0_full_matrix",),
    )
    changed = [dict(row) for row in rows]
    changed[1]["role"] = "gate"
    with pytest.raises(InvariantViolation, match="comparison role changed"):
        validate_comparison_partition(
            changed,
            gated_names=("hidden",),
            descriptive_names=("keep0_full_matrix",),
        )


def test_exact_failure_binding_is_byte_and_symlink_strict(tmp_path):
    failure = tmp_path / "failure.json"
    log = tmp_path / "run.log"
    failure.write_bytes(b"failure")
    log.write_bytes(b"log")
    validate_exact_failure_binding(
        failure_path=failure,
        log_path=log,
        expected_failure_sha256="16d34b5e7bcb341ee6cb3d16495d90e93fbe57c46d3827432613210a24ebca30",
        expected_log_sha256="836ff184e7b41b1e13cb5fd89fa1de98dbbab99e9d2918913ff43b86a5c7c213",
    )
    log.write_bytes(b"changed")
    with pytest.raises(InvariantViolation, match="hash changed"):
        validate_exact_failure_binding(
            failure_path=failure,
            log_path=log,
            expected_failure_sha256="16d34b5e7bcb341ee6cb3d16495d90e93fbe57c46d3827432613210a24ebca30",
            expected_log_sha256="836ff184e7b41b1e13cb5fd89fa1de98dbbab99e9d2918913ff43b86a5c7c213",
        )
