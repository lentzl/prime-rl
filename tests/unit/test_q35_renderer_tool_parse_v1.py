import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCHER_PATH = ROOT / "scripts/apply_q35_renderer_tool_parse_v1.py"
SPEC = importlib.util.spec_from_file_location("apply_q35_renderer_tool_parse_v1", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


def test_qwen35_complete_function_without_outer_close_is_executable() -> None:
    verifier = (ROOT / "scripts/verify_q35_renderer_tool_parse_v1.py").read_text()
    compile(verifier, "verify_q35_renderer_tool_parse_v1.py", "exec")
    assert "ToolCallParseStatus.OK" in verifier
    assert "ToolCallParseStatus.UNCLOSED_BLOCK" in verifier
    assert "reviewer = await rlm" in verifier


def test_qwen35_renderer_patcher_is_exact_and_idempotent(tmp_path: Path) -> None:
    parser_source = tmp_path / "parsing.py"
    parser_source.write_text("before\n" + PATCHER.OLD + "after\n")

    assert PATCHER.apply_parser_patch(parser_source) == "applied"
    assert parser_source.read_text() == "before\n" + PATCHER.NEW + "after\n"
    assert PATCHER.apply_parser_patch(parser_source) == "already_applied"


def test_role_launcher_applies_pinned_renderer_patch_before_gpu_launch() -> None:
    launcher = (ROOT / "scripts/run_q35_2b_role_grpo_v1.sh").read_text()
    patcher = PATCHER_PATH.read_text()

    apply_at = launcher.index('python "$renderer_patcher"')
    launch_at = launcher.index('rl @ "$resolved"')
    assert apply_at < launch_at
    assert "</function>\\\\s*$" in patcher
    assert "ToolCallParseStatus.UNCLOSED_BLOCK" in patcher
