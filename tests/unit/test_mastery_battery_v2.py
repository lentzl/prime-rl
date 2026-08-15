import os
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "qwen35-27b-prime-agent-mastery-v2"
CONFIG_NAMES = (
    "271-foundations.toml",
    "272-coordination-calibration.toml",
    "273-coordination-heldout.toml",
    "274-ownership-child.toml",
    "275-ownership-coordinator.toml",
    "276-oolong.toml",
)


def _load(name: str) -> dict:
    with (EXPERIMENT / name).open("rb") as stream:
        return tomllib.load(stream)


def test_mastery_v2_is_a_frozen_74_task_official_harness_battery() -> None:
    configs = [_load(name) for name in CONFIG_NAMES]

    assert sum(config["num_tasks"] for config in configs) == 74
    assert all(
        config["env"]["agent"]["harness"]
        == {
            "id": "prime_agent",
            "version": "0.7.2-beta.495.1.97b994c",
        }
        for config in configs
    )
    assert all(config["sampling"]["reasoning_effort"] == "high" for config in configs)
    assert all("thinking" not in config["env"]["agent"]["harness"] for config in configs)
    assert all("autonomous" not in config["env"]["agent"]["harness"] for config in configs)
    assert all("gates" not in config["env"]["agent"]["harness"] for config in configs)


def test_mastery_v2_preserves_the_historical_slice_boundaries() -> None:
    foundations, calibration, heldout, child, coordinator, oolong = (_load(name) for name in CONFIG_NAMES)

    assert foundations["env"]["taskset"] == {
        "id": "prime-agent-foundations-v2",
        "families": [
            "ipython_cell",
            "kernel_persistence",
            "conversation_resume",
            "child_result_delivery",
            "child_cancellation",
        ],
        "instances_per_family": 2,
        "instance_offset": 0,
    }
    assert calibration["env"]["taskset"]["split"] == "train"
    assert calibration["env"]["taskset"]["instances_per_template"] == 1
    assert heldout["env"]["taskset"]["split"] == "eval"
    assert heldout["env"]["taskset"]["instances_per_template"] == 2
    assert child["env"]["taskset"]["ownership"] == "child"
    assert coordinator["env"]["taskset"]["ownership"] == "coordinator"
    assert child["env"]["taskset"]["instance_offset"] == coordinator["env"]["taskset"]["instance_offset"]
    assert oolong["env"]["taskset"]["context_len"] == 16_384


def test_mastery_v2_launcher_records_the_exact_software_and_model_revisions() -> None:
    launcher = (ROOT / "scripts" / "run_qwen35_27b_prime_agent_mastery_battery_v2.sh").read_text()

    assert all(name.removesuffix(".toml") in launcher for name in CONFIG_NAMES)
    assert "git rev-parse HEAD" in launcher
    assert "git -C deps/verifiers rev-parse HEAD" in launcher
    assert "prime_agent_version=0.7.2-beta.495.1.97b994c" in launcher
    assert "fc05daec18b0a78c049392ed2e771dde82bdf654" in launcher
    assert "sha256sum" in launcher
    assert "EVAL_CLIENT_BASE_URL" in launcher
    assert '--run.name "$name"' in launcher
    assert '--run.dir "$name"' in launcher
    assert "summarize_prime_agent_mastery_v2.py" in launcher
    assert "--require-valid-traces" in launcher
    assert "expected_count" in launcher
    assert '--expected-count "$expected_count"' in launcher
    assert "SUMMARY.txt" in launcher
    assert "SUMMARY.json" in launcher


def test_mastery_v2_model_launcher_is_revision_pinned_and_fail_closed() -> None:
    launcher = (ROOT / "scripts" / "run_qwen35_27b_prime_agent_mastery_baseline_v2.sh").read_text()

    assert "fc05daec18b0a78c049392ed2e771dde82bdf654" in launcher
    assert "refusing to overwrite mastery output" in launcher
    assert "refusing to launch while another GPU process is active" in launcher
    assert "Prime Agent artifact is unavailable" in launcher
    assert "vllm-router is missing" in launcher
    assert "CUDA device count must equal tensor parallel size" in launcher
    assert '[[ ! -f "$model/STABLE" ]]' in launcher
    assert "BASELINE_DRY_RUN" in launcher
    assert 'mktemp -d "${TMPDIR:-/tmp}/qwen35-27b-mastery-v2.XXXXXX"' in launcher
    assert "serve_config=$preflight_dir/inference.toml" in launcher
    assert 'revision = "$revision"' in launcher
    assert "max_model_len = 65536" in launcher
    assert 'tool_call_parser = "qwen3_coder"' in launcher
    assert 'reasoning_parser = "qwen3"' in launcher
    assert "run_qwen35_27b_prime_agent_mastery_battery_v2.sh" in launcher
    assert "MODEL_REVISION=$revision" in launcher
    assert 'kill "$eval_pid"' in launcher
    assert 'kill "$inference_pid"' in launcher
    assert 'wait -n -p completed_pid "$inference_pid" "$eval_pid"' in launcher


def test_mastery_v2_dry_run_leaves_production_output_absent(tmp_path: Path) -> None:
    launcher = ROOT / "scripts" / "run_qwen35_27b_prime_agent_mastery_baseline_v2.sh"
    fake_inference = tmp_path / "inference"
    fake_eval = tmp_path / "eval"
    fake_nvidia_smi = tmp_path / "nvidia-smi"
    fake_router = tmp_path / "vllm-router"
    fake_curl = tmp_path / "curl"
    seen_config = tmp_path / "seen-config"
    fake_inference.write_text(
        '#!/bin/sh\ntest "$1" = @\ntest -f "$2"\ncase "$2" in *.toml) ;; *) exit 1 ;; esac\n'
        'printf "%s" "$2" > "$SEEN_CONFIG"\n'
    )
    fake_eval.write_text("#!/bin/sh\nexit 0\n")
    fake_nvidia_smi.write_text("#!/bin/sh\nexit 0\n")
    fake_router.write_text("#!/bin/sh\nexit 0\n")
    fake_curl.write_text("#!/bin/sh\nprintf '%s  %s\\n' deadbeef prime-agent-0.7.2-beta.495.1.97b994c.tgz\n")
    for executable in (
        fake_inference,
        fake_eval,
        fake_nvidia_smi,
        fake_router,
        fake_curl,
    ):
        executable.chmod(0o755)

    output_root = tmp_path / "production"
    env = os.environ | {
        "BASELINE_DRY_RUN": "true",
        "PRIME_MASTERY_OUTPUT_ROOT": str(output_root),
        "INFERENCE_BIN": str(fake_inference),
        "EVAL_BIN": str(fake_eval),
        "NVIDIA_SMI_BIN": str(fake_nvidia_smi),
        "VLLM_ROUTER_BIN": str(fake_router),
        "CURL_BIN": str(fake_curl),
        "SEEN_CONFIG": str(seen_config),
    }
    subprocess.run(
        [str(launcher), "Qwen/Qwen3.5-27B", "untouched-base"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    temporary_config = Path(seen_config.read_text())
    assert not temporary_config.exists()
    assert not temporary_config.parent.exists()
    assert not (output_root / "untouched-base").exists()
