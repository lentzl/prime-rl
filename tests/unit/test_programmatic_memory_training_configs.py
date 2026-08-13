from pathlib import Path

ROOT = Path(__file__).parents[2]
SFT = ROOT / "configs/debug/subagent-communication/340-qwen35-27b-memory-v2-plain-sft.toml"
SDFT = ROOT / "configs/debug/subagent-communication/341-qwen35-27b-memory-v2-sdft.toml"
LAUNCH = ROOT / "scripts/run_programmatic_memory_v2_training.sh"


def test_plain_sft_is_full_weight_and_four_pass_budgeted() -> None:
    source = SFT.read_text()

    assert "max_steps = 154" in source
    assert "name = \"/ephemeral/subagent-rung/data/programmatic-episodic-memory-v2/train.parquet\"" in source
    assert "assistant = true" in source
    assert "[model.lora]" not in source


def test_sdft_is_native_opsd_with_one_rollout_per_example() -> None:
    source = SDFT.read_text()

    assert "max_steps = 200" in source
    assert "batch_size = 24" in source
    assert "group_size = 1" in source
    assert "type = \"opsd\"" in source
    assert "condition_on_demonstration = false" in source
    assert "[trainer.model.lora]" not in source


def test_launcher_enforces_admission_only_for_sdft() -> None:
    source = LAUNCH.read_text()

    assert "--require-pass" in source
    assert 'case "$lane" in' in source
    assert "refusing to launch while another GPU process is active" in source
    assert 'command=(sft @ "$config" --model.name "$start_model")' in source
    assert 'command=(rl @ "$config" --model.name "$start_model")' in source
