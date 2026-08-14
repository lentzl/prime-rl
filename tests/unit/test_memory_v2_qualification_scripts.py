from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_tranche_qualification_freezes_base_and_selected_checkpoints() -> None:
    launcher = (
        ROOT
        / "scripts"
        / "run_qwen35_27b_memory_v2_tranche_qualification_v1.sh"
    ).read_text()

    assert "fc05daec18b0a78c049392ed2e771dde82bdf654" in launcher
    assert "steps=(1 2 4 8)" in launcher
    assert '[[ ! -f "$checkpoint/STABLE" ]]' in launcher
    assert "labels=(base step-1 step-2 step-4 step-8)" in launcher
    assert "refusing to mix partial qualification output" in launcher
    assert '[[ -f "$run_output/QUALIFICATION_COMPLETE" ]]' in launcher


def test_combined_driver_runs_memory_and_mastery_before_marking_complete() -> None:
    driver = (
        ROOT / "scripts" / "run_qwen35_27b_memory_v2_combined_qualification.sh"
    ).read_text()

    memory = driver.index("run_qwen35_27b_memory_v2_full_eval.sh")
    memory_summary = driver.index("summarize_programmatic_memory_eval.py")
    mastery = driver.index("run_qwen35_27b_mastery_battery_v1.sh")
    mastery_summary = driver.index("summarize_prime_agent_mastery.py")
    complete = driver.index("QUALIFICATION_COMPLETE", driver.index("printf"))

    assert memory < memory_summary < mastery < mastery_summary < complete
    assert "mastery battery produced no traces" in driver
    assert "--json --summary-only" in driver
