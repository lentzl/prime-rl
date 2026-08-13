from pathlib import Path


def test_host_wide_process_cleanup_is_ci_only() -> None:
    source = (Path(__file__).parents[1] / "conftest.py").read_text()

    guard = source.index('if os.environ.get("CI"):')
    assert guard < source.index('subprocess.run(["pkill", "-f", "torchrun"])')
    assert guard < source.index('subprocess.run(["pkill", "-f", "VLLM"])')
