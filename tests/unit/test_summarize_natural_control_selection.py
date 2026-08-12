import json
from pathlib import Path

from scripts.summarize_natural_control_selection import summarize


def _trace(index: int, family: str, *, path_access: float = 0.0, aligned: float = 1.0) -> dict:
    return {
        "id": f"{family}-{index}",
        "task": {"data": {"idx": index, "family": family, "template_variant": index}},
        "metrics": {
            "answer_accuracy": 1.0,
            "natural_followup_causal": 1.0,
            "protocol_aligned": aligned,
            "clean_protocol_aligned": 0.0,
            "coordinator_delegated_path_accesses": path_access,
            "bidirectional_control": aligned,
        },
    }


def _write(path: Path, traces: list[dict]) -> None:
    path.write_text("".join(json.dumps({"traces": [trace]}) + "\n" for trace in traces))


def test_natural_gate_preserves_each_family(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    traces = [_trace(0, "followup"), _trace(1, "handshake")]
    _write(base, traces)
    _write(candidate, traces)

    assert summarize(base, candidate)["promotion_pass"] is True


def test_natural_gate_rejects_path_access_even_when_scores_hold(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(base, [_trace(0, "followup"), _trace(1, "handshake")])
    _write(candidate, [_trace(0, "followup", path_access=1.0), _trace(1, "handshake")])

    report = summarize(base, candidate)

    assert report["family_passes"] == {"followup": True, "handshake": True}
    assert report["promotion_pass"] is False
