from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/finish_prime_agent_teacher_bootstrap.sh"
TRAINER_SCRIPT = ROOT / "scripts/run_prime_agent_teacher_bootstrap_online.sh"
SETUP_SCRIPT = ROOT / "scripts/setup_prime_agent_mastery_host.sh"


def test_finish_script_uses_only_final_verified_sources() -> None:
    source = SCRIPT.read_text()

    assert "278-qwen35-27b-mastery-child-teacher-collection" in source
    assert "290-qwen35-27b-mastery-guided-child-ownership-supplement" in source
    assert "279-qwen35-27b-mastery-coordinator-teacher-collection" in source
    assert "288-qwen35-27b-mastery-guided-coordinator-ownership-supplement" in source
    assert "280-qwen35-27b-mastery-guided-communication-collection" in source
    assert "292-qwen35-27b-mastery-hard-communication-supplement" in source
    assert "308-qwen35-27b-mastery-evidence-gated-handshake-supplement" in source
    assert "314-qwen35-27b-mastery-literal-safe-followup-supplement" in source
    for excluded in ("302-qwen", "304-qwen", "306-qwen", "310-qwen", "312-qwen"):
        assert excluded not in source


def test_finish_script_is_fail_closed_before_training() -> None:
    source = SCRIPT.read_text()

    assert "set -euo pipefail" in source
    assert 'metrics.get("clean_protocol_aligned") == 1' in source
    assert 'metrics.get("bidirectional_control") == 1' in source
    assert "AUDIT_ONLY=1" in source
    assert "refusing to overwrite existing bootstrap dataset" in source
    assert "GPU processes did not quiesce" in source
    assert "run_prime_agent_teacher_bootstrap_online.sh" in source


def test_trainer_launcher_exposes_venv_sibling_commands() -> None:
    source = TRAINER_SCRIPT.read_text()

    assert 'export PATH="$root/.venv/bin:$PATH"' in source
    assert "vllm-router is missing" in source
    assert 'exec .venv/bin/sft @ "$config"' in source


def test_host_setup_installs_only_the_pinned_router_wheel() -> None:
    source = SETUP_SCRIPT.read_text()

    assert source.count("vllm_router-0.1.26-") == 2
    assert '"$uv" pip install --python .venv/bin/python "$router_wheel"' in source
    assert "--extra disagg" not in source
    assert "test -x .venv/bin/vllm-router" in source
