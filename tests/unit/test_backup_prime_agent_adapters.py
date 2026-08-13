import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "backup_prime_agent_adapters.py"
SPEC = importlib.util.spec_from_file_location("backup_prime_agent_adapters", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_only_complete_stable_adapters_are_discovered(tmp_path: Path) -> None:
    complete = tmp_path / "weights" / "step_2"
    incomplete = tmp_path / "weights" / "step_4"
    for checkpoint in (complete, incomplete):
        adapter = checkpoint / "lora_adapters"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (complete / "STABLE").touch()

    assert MODULE.stable_adapters(tmp_path) == [(2, complete / "lora_adapters")]
