import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")
pytest.importorskip("verifiers")

from scripts import export_sft


def test_main_reads_current_episode_envelopes(monkeypatch, tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "traces.jsonl").write_text(
        json.dumps({"id": "episode", "env": {"id": "test"}, "ok": True, "errors": [], "traces": []}) + "\n"
    )
    trace = SimpleNamespace(
        reward=1.0,
        stop_condition="agent_completed",
        is_truncated=False,
        metrics={"strict_success": 1.0},
    )
    episode = SimpleNamespace(traces=[trace])
    monkeypatch.setattr(export_sft, "read_episodes", lambda *_: [episode])
    monkeypatch.setattr(export_sft, "sft_rows", lambda _: [{"messages": [], "tools": "[]"}])
    monkeypatch.setattr(export_sft.Dataset, "from_list", lambda rows: SimpleNamespace(to_parquet=lambda _: None))
    monkeypatch.setattr(
        "sys.argv",
        ["export_sft.py", str(run_dir), "--min-reward", "1.0"],
    )

    export_sft.main()

    assert "1 episode(s), 1 trace(s) -> 1 row(s)" in capsys.readouterr().out


def test_keep_requires_explicit_saved_metrics() -> None:
    accepted = SimpleNamespace(
        reward=3.0,
        stop_condition="agent_completed",
        is_truncated=False,
        metrics={"answer_accuracy": 1.0, "clean_protocol_aligned": 1.0},
    )
    missing = SimpleNamespace(**{**accepted.__dict__, "metrics": {"answer_accuracy": 1.0}})

    requirements = [
        export_sft.parse_metric_requirement("answer_accuracy=1"),
        export_sft.parse_metric_requirement("clean_protocol_aligned=1"),
    ]
    assert export_sft.keep(accepted, None, False, requirements)
    assert not export_sft.keep(missing, None, False, requirements)
