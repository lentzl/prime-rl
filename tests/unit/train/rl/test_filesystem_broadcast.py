from types import SimpleNamespace

import pytest

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig
from prime_rl.trainer.rl.broadcast import filesystem, setup_weight_broadcast
from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast, extra_model_save_dir


class NamedModel:
    def __init__(self, name: str):
        self.name = name


class FakeMultiRunManager:
    def __init__(self, run_dir):
        self.ready_to_update_idxs = [0]
        self.ready_to_update = {0: True}
        self.progress = {0: SimpleNamespace(step=7)}
        self.idx_2_id = {0: "run-0"}
        self.run_dir = run_dir

    def get_run_dir(self, idx):
        assert idx == 0
        return self.run_dir

    def get_orchestrator_config(self, run_id):
        assert run_id == "run-0"
        return object()


def test_extra_model_save_dir_rejects_unsafe_roles(tmp_path):
    assert extra_model_save_dir(tmp_path, "sdpo_teacher") == tmp_path / "sdpo_teacher"

    for role in ["", ".", "..", "nested/path", "/absolute"]:
        with pytest.raises(ValueError, match="invalid extra model role"):
            extra_model_save_dir(tmp_path, role)


def test_setup_weight_broadcast_constructs_filesystem_without_vllm(monkeypatch, tmp_path):
    monkeypatch.setattr(filesystem, "get_world", lambda: SimpleNamespace(is_master=True))
    monkeypatch.setattr(filesystem, "get_multi_run_manager", lambda: FakeMultiRunManager(tmp_path))

    broadcast = setup_weight_broadcast(tmp_path, FileSystemWeightBroadcastConfig())

    assert isinstance(broadcast, FileSystemWeightBroadcast)


def test_filesystem_broadcast_saves_extra_models_before_stable(monkeypatch, tmp_path):
    saves = []

    def fake_gather_hf_state_dict(model, *, is_master):
        assert is_master
        return {f"{model.name}.weight": object()}

    def fake_save_state_dict(state_dict, save_dir, save_format, save_sharded, adapter=False):
        step_dir = tmp_path / "broadcasts" / "step_7"
        saves.append(
            {
                "keys": tuple(state_dict.keys()),
                "save_dir": save_dir,
                "root_stable_exists": (step_dir / "STABLE").exists(),
                "teacher_stable_exists": (step_dir / "sdpo_teacher" / "STABLE").exists(),
                "adapter": adapter,
            }
        )
        state_dict.clear()

    monkeypatch.setattr(filesystem, "get_world", lambda: SimpleNamespace(is_master=True))
    monkeypatch.setattr(filesystem, "get_multi_run_manager", lambda: FakeMultiRunManager(tmp_path))
    monkeypatch.setattr(filesystem, "gather_hf_state_dict", fake_gather_hf_state_dict)
    monkeypatch.setattr(filesystem, "save_state_dict", fake_save_state_dict)

    broadcast = FileSystemWeightBroadcast(tmp_path, FileSystemWeightBroadcastConfig())
    broadcast.broadcast_weights(NamedModel("policy"), step=7, extra_models={"sdpo_teacher": NamedModel("teacher")})

    step_dir = tmp_path / "broadcasts" / "step_7"
    assert saves == [
        {
            "keys": ("policy.weight",),
            "save_dir": step_dir,
            "root_stable_exists": False,
            "teacher_stable_exists": False,
            "adapter": False,
        },
        {
            "keys": ("teacher.weight",),
            "save_dir": step_dir / "sdpo_teacher",
            "root_stable_exists": False,
            "teacher_stable_exists": False,
            "adapter": False,
        },
    ]
    assert (step_dir / "sdpo_teacher" / "STABLE").exists()
    assert (step_dir / "STABLE").exists()
    assert not broadcast.multi_run_manager.ready_to_update[0]
