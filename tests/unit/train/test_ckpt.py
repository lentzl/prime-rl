import pytest

from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer import ckpt
from prime_rl.trainer.ckpt import AppState, CheckpointManager


class ExtraState:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"value": 3}

    def load_state_dict(self, state_dict):
        self.loaded = state_dict


class RequiredExtraState(ExtraState):
    requires_checkpoint_state = True


def test_app_state_saves_extra_state(monkeypatch):
    monkeypatch.setattr(ckpt, "get_state_dict", lambda model, optimizers: ({"model": 1}, {"optim": 2}))
    extra_state = ExtraState()

    state_dict = AppState(
        model=object(),
        optimizers=[],
        scheduler=None,
        progress=None,
        extra_state=extra_state,
    ).state_dict()

    assert state_dict["model"] == {"model": 1}
    assert state_dict["optimizers"] == {"optim": 2}
    assert state_dict["extra_state"] == {"value": 3}


def test_app_state_loads_extra_state(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ckpt,
        "set_state_dict",
        lambda model, optimizers, *, model_state_dict, optim_state_dict: calls.append(
            (model_state_dict, optim_state_dict)
        ),
    )
    extra_state = ExtraState()
    app_state = AppState(
        model=object(),
        optimizers=[],
        scheduler=None,
        progress=None,
        extra_state=extra_state,
    )

    app_state.load_state_dict(
        {
            "model": {"model": 1},
            "optimizers": {"optim": 2},
            "extra_state": {"value": 4},
        }
    )

    assert calls == [({"model": 1}, {"optim": 2})]
    assert extra_state.loaded == {"value": 4}


def test_app_state_loads_old_checkpoints_without_extra_state(monkeypatch):
    monkeypatch.setattr(
        ckpt,
        "set_state_dict",
        lambda model, optimizers, *, model_state_dict, optim_state_dict: None,
    )
    extra_state = ExtraState()
    app_state = AppState(
        model=object(),
        optimizers=[],
        scheduler=None,
        progress=None,
        extra_state=extra_state,
    )

    app_state.load_state_dict({"model": {}, "optimizers": {}})

    assert extra_state.loaded is None


def test_app_state_rejects_missing_required_extra_state(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ckpt,
        "set_state_dict",
        lambda model, optimizers, *, model_state_dict, optim_state_dict: calls.append(
            (model_state_dict, optim_state_dict)
        ),
    )
    extra_state = RequiredExtraState()
    app_state = AppState(
        model=object(),
        optimizers=[],
        scheduler=None,
        progress=None,
        extra_state=extra_state,
    )

    with pytest.raises(ValueError, match="RequiredExtraState requires checkpoint extra_state"):
        app_state.load_state_dict({"model": {}, "optimizers": {}})

    assert calls == []
    assert extra_state.loaded is None


def test_checkpoint_manager_marks_step_stable_after_save(monkeypatch, tmp_path):
    events = []

    monkeypatch.setattr(ckpt, "get_world", lambda: type("World", (), {"is_master": True})())
    monkeypatch.setattr(ckpt.torch.distributed, "barrier", lambda: events.append(("barrier",)))

    def fake_save_to_path(self, path, model, optimizers, scheduler, progress, dataloader=None, extra_state=None):
        events.append(("save", path, (path.parent / "STABLE").exists()))

    monkeypatch.setattr(CheckpointManager, "save_to_path", fake_save_to_path)

    manager = CheckpointManager(tmp_path, CheckpointConfig())
    manager.save(step=3, model=object(), optimizers=[], scheduler=None, progress=object())

    step_dir = tmp_path / "checkpoints" / "step_3"
    assert events == [("barrier",), ("save", step_dir / "trainer", False)]
    assert (step_dir / "STABLE").exists()
    assert manager.ckpt_steps == [3]
