from pathlib import Path

import pytest

from scripts.audit_native_sibling_sdpo_run import _load_config, _structural_pass


def _write_config(path: Path, learning_rate: float) -> None:
    path.write_text(
        f"""
[orchestrator.algo]
type = "sdpo"
dont_reprompt_on_self_success = true
include_environment_feedback = false
multi_turn_replay = false

[trainer.optim]
lr = {learning_rate}
""".strip()
        + "\n"
    )


def test_load_config_accepts_declared_nonzero_learning_rate(tmp_path: Path) -> None:
    path = tmp_path / "dose.toml"
    _write_config(path, 1e-7)

    config = _load_config(path, expected_learning_rate=1e-7)

    assert config["trainer"]["optim"]["lr"] == 1e-7


def test_load_config_rejects_unexpected_learning_rate(tmp_path: Path) -> None:
    path = tmp_path / "dose.toml"
    _write_config(path, 1e-7)

    with pytest.raises(ValueError, match="expected trainer.optim.lr=0.0, found 1e-07"):
        _load_config(path, expected_learning_rate=0.0)


def test_structural_pass_can_reuse_prior_no_success_control() -> None:
    assert _structural_pass(
        replay_groups=5,
        no_success_groups=0,
        exported_branches=78,
        require_no_success_group=False,
    )
    assert not _structural_pass(
        replay_groups=5,
        no_success_groups=0,
        exported_branches=78,
        require_no_success_group=True,
    )
