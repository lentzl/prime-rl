import os
import subprocess
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


def test_tranche_qualification_uses_disjoint_gpu_groups_without_changing_battery() -> None:
    launcher = (
        ROOT
        / "scripts"
        / "run_qwen35_27b_memory_v2_tranche_qualification_v1.sh"
    ).read_text()

    assert "MEMORY_QUALIFICATION_PARALLELISM:-2" in launcher
    assert "device_groups=(0,1,2,3 4,5,6,7)" in launcher
    assert "EVAL_TENSOR_PARALLEL_SIZE=4" in launcher
    assert "EVAL_CUDA_VISIBLE_DEVICES=${device_groups[$slot]}" in launcher
    assert "EVAL_BACKEND_PORT=$backend_port" in launcher
    assert "EVAL_ROUTER_PORT=$router_port" in launcher
    assert "EVAL_DATA_PARALLEL_RPC_PORT=$rpc_port" in launcher
    assert "run_qwen35_27b_memory_v2_combined_qualification.sh" in launcher


def test_tranche_qualification_dispatches_every_model_to_isolated_slots(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    for step in (1, 2, 4, 8):
        checkpoint = source / "weights" / f"step_{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "STABLE").touch()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)

    log = tmp_path / "launches.txt"
    model_launcher = tmp_path / "model-launcher.sh"
    model_launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s|%s|%s|%s|%s\\n" "$2" "$EVAL_CUDA_VISIBLE_DEVICES" '
        '"$EVAL_BACKEND_PORT" "$EVAL_ROUTER_PORT" "$EVAL_DATA_PARALLEL_RPC_PORT" '
        '>>"$QUALIFICATION_TEST_LOG"\n'
        'mkdir -p "$PRIME_MASTERY_OUTPUT_ROOT/$2"\n'
        'touch "$PRIME_MASTERY_OUTPUT_ROOT/$2/QUALIFICATION_COMPLETE"\n',
        encoding="utf-8",
    )
    model_launcher.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "MEMORY_TRANCHE_OUTPUT": str(source),
            "MEMORY_QUALIFICATION_OUTPUT": str(output),
            "MEMORY_QUALIFICATION_PARALLELISM": "2",
            "MEMORY_QUALIFICATION_MODEL_LAUNCHER": str(model_launcher),
            "QUALIFICATION_TEST_LOG": str(log),
        }
    )
    subprocess.run(
        [str(ROOT / "scripts" / "run_qwen35_27b_memory_v2_tranche_qualification_v1.sh")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    launches = {
        fields[0]: fields[1:]
        for line in log.read_text(encoding="utf-8").splitlines()
        if (fields := line.split("|"))
    }
    assert launches == {
        "base": ["0,1,2,3", "8100", "8000", "13345"],
        "step-1": ["4,5,6,7", "8200", "8100", "13445"],
        "step-2": ["0,1,2,3", "8100", "8000", "13345"],
        "step-4": ["4,5,6,7", "8200", "8100", "13445"],
        "step-8": ["0,1,2,3", "8100", "8000", "13345"],
    }
    assert all((output / label / "QUALIFICATION_COMPLETE").is_file() for label in launches)


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
