import importlib
import sys
import tomllib
from types import SimpleNamespace

import pytest

from prime_rl.configs.rl import MultiNodeDeploymentConfig, RLConfig

MODEL_NAME = "Qwen/Qwen3-4B-Instruct"


def _load_rl_entrypoint(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        SimpleNamespace(nvmlInit=lambda: None, nvmlDeviceGetCount=lambda: 8),
    )
    return importlib.import_module("prime_rl.entrypoints.rl")


def _sdpo_auto_teacher_config(tmp_path, *, inference_overrides=None):
    inference = {
        "model": {"name": MODEL_NAME},
        "server": {"port": 8100},
        "parallel": {"tp": 2, "dp": 1},
        "api_server_count": 1,
        "data_parallel_rpc_port": 14000,
    }
    if inference_overrides:
        for key, value in inference_overrides.items():
            if isinstance(value, dict) and isinstance(inference.get(key), dict):
                inference[key].update(value)
            else:
                inference[key] = value
    return RLConfig.model_validate(
        {
            "output_dir": tmp_path,
            "trainer": {"model": {"name": MODEL_NAME}},
            "inference": inference,
            "deployment": {
                "gpus_per_node": 8,
                "num_train_gpus": 1,
                "num_infer_gpus": 2,
                "num_sdpo_teacher_gpus": 4,
            },
            "orchestrator": {
                "renderer": {"name": "qwen3"},
                "model": {"name": MODEL_NAME},
                "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                "train": {"env": [{"id": "reverse-text"}]},
            },
        }
    )


def test_make_sdpo_teacher_inference_config_clones_policy_inference_for_auto_launch(tmp_path, monkeypatch):
    rl_entrypoint = _load_rl_entrypoint(monkeypatch)
    config = _sdpo_auto_teacher_config(tmp_path)

    teacher = rl_entrypoint.make_sdpo_teacher_inference_config(config)

    assert teacher.model.name == MODEL_NAME
    assert teacher.model.trust_remote_code == config.orchestrator.model.trust_remote_code
    assert teacher.server.port == 8101
    assert teacher.data_parallel_rpc_port == 15000
    assert teacher.parallel.tp == 2
    assert teacher.parallel.dp == 2
    assert teacher.api_server_count == 2

    # The policy inference config is cloned, not mutated, so the main server
    # stays on its own port/parallelism.
    assert config.inference is not None
    assert config.inference.server.port == 8100
    assert config.inference.parallel.dp == 1
    assert config.inference.api_server_count == 1


def test_make_sdpo_teacher_inference_config_resyncs_local_dp_after_resize(tmp_path, monkeypatch):
    rl_entrypoint = _load_rl_entrypoint(monkeypatch)
    config = _sdpo_auto_teacher_config(tmp_path, inference_overrides={"data_parallel_size_local": 1})

    teacher = rl_entrypoint.make_sdpo_teacher_inference_config(config)

    assert teacher.parallel.dp == 2
    assert teacher.data_parallel_size_local == 2


def test_make_sdpo_teacher_inference_config_rejects_rpc_port_offset_overflow(tmp_path, monkeypatch):
    rl_entrypoint = _load_rl_entrypoint(monkeypatch)
    config = _sdpo_auto_teacher_config(tmp_path, inference_overrides={"data_parallel_rpc_port": 65000})

    with pytest.raises(ValueError, match="RPC offset"):
        rl_entrypoint.make_sdpo_teacher_inference_config(config)


def test_make_sdpo_teacher_inference_config_requires_inference_config(tmp_path, monkeypatch):
    rl_entrypoint = _load_rl_entrypoint(monkeypatch)
    config = _sdpo_auto_teacher_config(tmp_path)
    config.inference = None

    with pytest.raises(ValueError, match=r"requires an \[inference\] config"):
        rl_entrypoint.make_sdpo_teacher_inference_config(config)


def test_make_sdpo_teacher_inference_config_requires_teacher_endpoint(tmp_path, monkeypatch):
    rl_entrypoint = _load_rl_entrypoint(monkeypatch)
    config = _sdpo_auto_teacher_config(tmp_path)
    config.orchestrator.sdpo_teacher = None

    with pytest.raises(ValueError, match="requires orchestrator.sdpo_teacher"):
        rl_entrypoint.make_sdpo_teacher_inference_config(config)


def test_write_subconfigs_writes_separate_sdpo_teacher_inference_toml(tmp_path, monkeypatch):
    rl_entrypoint = _load_rl_entrypoint(monkeypatch)
    config = _sdpo_auto_teacher_config(tmp_path)
    config_dir = tmp_path / "configs"

    rl_entrypoint.write_subconfigs(config, config_dir)

    teacher_toml = config_dir / rl_entrypoint.SDPO_TEACHER_INFERENCE_TOML
    assert teacher_toml.exists()
    with teacher_toml.open("rb") as f:
        teacher_config = tomllib.load(f)

    assert teacher_config["model"]["name"] == MODEL_NAME
    assert teacher_config["server"]["port"] == 8101
    assert teacher_config["parallel"] == {"tp": 2, "dp": 2}
    assert teacher_config["api_server_count"] == 2
    assert teacher_config["data_parallel_rpc_port"] == 15000

    with (config_dir / rl_entrypoint.ORCHESTRATOR_TOML).open("rb") as f:
        orchestrator_config = tomllib.load(f)

    assert orchestrator_config["sdpo_teacher"]["client"]["extra_headers_from_state"] == {
        "X-Session-ID": "trajectory_id"
    }


def test_rl_local_rejects_non_single_node_deployment(tmp_path, monkeypatch):
    rl_entrypoint = _load_rl_entrypoint(monkeypatch)
    config = RLConfig.model_validate({"output_dir": tmp_path, "trainer": {}, "orchestrator": {}})
    config.deployment = MultiNodeDeploymentConfig.model_validate({"num_train_nodes": 1, "num_infer_nodes": 0})

    with pytest.raises(ValueError, match="rl_local only supports"):
        rl_entrypoint.rl_local(config)


def test_write_slurm_script_requires_slurm_config(tmp_path, monkeypatch):
    rl_entrypoint = _load_rl_entrypoint(monkeypatch)
    config = RLConfig.model_validate({"output_dir": tmp_path, "trainer": {}, "orchestrator": {}})

    with pytest.raises(ValueError, match="requires config.slurm"):
        rl_entrypoint.write_slurm_script(config, tmp_path / "configs", tmp_path / "rl.sbatch")


def test_write_slurm_script_requires_slurm_template_path(tmp_path, monkeypatch):
    rl_entrypoint = _load_rl_entrypoint(monkeypatch)
    config = RLConfig.model_validate({"output_dir": tmp_path, "trainer": {}, "orchestrator": {}, "slurm": {}})
    config.slurm.template_path = None

    with pytest.raises(ValueError, match="requires config.slurm.template_path"):
        rl_entrypoint.write_slurm_script(config, tmp_path / "configs", tmp_path / "rl.sbatch")


def test_rl_slurm_requires_slurm_config(tmp_path, monkeypatch):
    rl_entrypoint = _load_rl_entrypoint(monkeypatch)
    config = RLConfig.model_validate({"output_dir": tmp_path, "trainer": {}, "orchestrator": {}})

    with pytest.raises(ValueError, match="rl_slurm requires config.slurm"):
        rl_entrypoint.rl_slurm(config)
