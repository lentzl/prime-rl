import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "validate_natural_yield_sdpo_zero_lr_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_natural_yield_sdpo_zero_lr_v1", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_exports(run_dir: Path, records: list[dict]) -> None:
    export_dir = run_dir / "token_exports" / "step_1"
    export_dir.mkdir(parents=True)
    (export_dir / "rank_0.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    (export_dir / "STABLE").touch()


def _export(token_ids: list[int], loss_mask: list[bool], sdpo: list[float]) -> dict:
    length = len(token_ids)
    return {
        "schema_version": 1,
        "step": 1,
        "env_name": MODULE.ENV_NAME,
        "token_ids": token_ids,
        "loss_mask": loss_mask,
        "rl_weights": [0.0] * length,
        "ce_weights": [0.0] * length,
        "ref_kl_weights": [0.0] * length,
        "sdpo_weights": sdpo,
    }


def test_token_routing_mirrors_trainer_sequence_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    limit = MODULE.TRAINING_SEQ_LEN
    active_tokens = list(range(limit + 2))
    active_loss = [False] * limit + [True, True]
    active_loss[limit - 2 : limit] = [True, True]
    active_sdpo = [False] * (limit - 2) + [True, True, False, False]
    dropped_tokens = list(range(20_000, 20_000 + limit + 1))
    dropped_loss = [False] * limit + [True]
    dropped_sdpo = [False] * (limit + 1)
    child_tokens = [30_000, 30_001]
    child_loss = [True, True]
    child_sdpo = [False, False]

    branches = [
        (SimpleNamespace(token_ids=active_tokens), active_loss),
        (SimpleNamespace(token_ids=dropped_tokens), dropped_loss),
        (SimpleNamespace(token_ids=child_tokens), child_loss),
    ]
    trace = SimpleNamespace(id="trace", branches=branches)
    monkeypatch.setattr(MODULE, "iter_trainable_branches", lambda value: value.branches)
    monkeypatch.setattr(
        MODULE,
        "keep_natural_yield_feedback_response",
        lambda value: [active_sdpo, dropped_sdpo, child_sdpo],
    )
    monkeypatch.setattr(
        MODULE,
        "_is_child_branch",
        lambda branch: branch.token_ids is child_tokens,
    )
    monkeypatch.setattr(MODULE, "EXPECTED_BATCH_SIZE", 1)

    _write_exports(
        tmp_path,
        [
            _export(
                active_tokens[:limit],
                active_loss[:limit],
                [float(value) for value in active_sdpo[:limit]],
            ),
            _export(child_tokens, child_loss, [0.0, 0.0]),
        ],
    )

    assert MODULE._validate_token_routing(tmp_path, [trace]) == {
        "export_records": 2,
        "coordinator_active_samples": 1,
        "child_zero_sdpo_samples": 1,
    }


def test_token_routing_rejects_wrong_truncated_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    limit = MODULE.TRAINING_SEQ_LEN
    token_ids = list(range(limit + 1))
    loss_mask = [False] * (limit - 1) + [True, False]
    expected = [False] * (limit - 1) + [True, False]
    trace = SimpleNamespace(
        id="trace",
        branches=[(SimpleNamespace(token_ids=token_ids), loss_mask)],
    )
    monkeypatch.setattr(MODULE, "iter_trainable_branches", lambda value: value.branches)
    monkeypatch.setattr(
        MODULE, "keep_natural_yield_feedback_response", lambda value: [expected]
    )
    monkeypatch.setattr(MODULE, "_is_child_branch", lambda branch: False)

    wrong_prefix = token_ids[:limit]
    wrong_prefix[-1] = -1
    _write_exports(
        tmp_path,
        [
            _export(
                wrong_prefix,
                loss_mask[:limit],
                [float(value) for value in expected[:limit]],
            )
        ],
    )

    with pytest.raises(MODULE.AuditFailure, match="no token export matches"):
        MODULE._validate_token_routing(tmp_path, [trace])
