from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_teacher_candidate_sequence_is_fail_closed_and_complete() -> None:
    script = (
        ROOT / "scripts" / "run_qwen35_27b_prime_agent_teacher_candidate_v1.sh"
    ).read_text()

    assert "untouched mastery baseline is incomplete" in script
    assert "while a GPU process is active" in script
    assert ".venv/bin/ruff check" in script
    assert ".venv/bin/pytest -q" in script
    assert "run_qwen35_27b_prime_agent_resilience_v1.sh" in script
    assert "run_qwen35_27b_prime_agent_sdpo_zero_lr_audit_v1.sh" in script
    assert "SDPO_MINIMUM_UPDATE_DRY_RUN=true" in script
    assert "run_qwen35_27b_prime_agent_sdpo_minimum_update_v1.sh" in script
    assert script.count("run_qwen35_27b_prime_agent_mastery_baseline_v2.sh") == 3
    assert "zero-LR audit report is not a matching pass" in script
    assert "validate_prime_agent_sdpo_minimum_update_v1.py" in script
    assert "compare_prime_agent_teacher_candidate_v1.py" in script
    assert "COMPARISON.json" in script
    assert "COMPLETE.txt" in script
