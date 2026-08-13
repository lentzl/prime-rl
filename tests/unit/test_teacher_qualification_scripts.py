from pathlib import Path

ROOT = Path(__file__).parents[2]
PREPARE = ROOT / "scripts/prepare_prime_agent_teacher_qualification.py"
LAUNCH = ROOT / "scripts/run_prime_agent_teacher_qualification.sh"


def test_preparer_keeps_frozen_run_and_repairs_only_eval_capacity() -> None:
    source = PREPARE.read_text()

    assert 'raw["ckpt"]["output_dir"] = str(source_output)' in source
    assert 'raw["deployment"]["num_gpus"] = 0' in source
    assert 'raw["inference"]["vllm"]["max_model_len"] = max_model_len' in source
    assert 'raw["inference"]["vllm"]["data_parallel_size"] = data_parallel_size' in source
    assert 'source["num_examples"] = oolong_examples' in source
    assert "write_eval_subconfigs(config" in source


def test_launcher_is_fresh_gpu_scoped_and_process_scoped() -> None:
    source = LAUNCH.read_text()

    assert "set -euo pipefail" in source
    assert "refusing to overwrite qualification output" in source
    assert "refusing to launch while another GPU process is active" in source
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3 inference" in source
    assert 'kill "${pids[$index]}"' in source
    assert "pkill" not in source
    assert 'wait "$evaluator_pid"' in source
