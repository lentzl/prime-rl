import json
from pathlib import Path

from scripts.summarize_native_sibling_continuation import summarize


def _write_step(
    root: Path,
    step: int,
    *,
    gains: int,
    followup_causal: int,
    handshake_bidirectional: float = 0.9,
    child_path: int = 0,
) -> None:
    path = root / f"step-{step}"
    path.mkdir()
    ownership = {
        "promotion_pass": gains >= 6,
        "strict_gains": gains,
        "strict_losses": 0,
        "candidate_child_path_accesses": child_path,
        "candidate_direct_spawns": 0,
    }
    natural = {
        "promotion_pass": False,
        "candidate": {
            "followup": {
                "answer_accuracy": 4,
                "natural_followup_causal": followup_causal,
                "protocol_aligned": 1,
                "bidirectional_control_mean": 0.4,
                "coordinator_delegated_path_accesses": 4,
            },
            "handshake": {
                "answer_accuracy": 4,
                "natural_followup_causal": 4,
                "protocol_aligned": 4,
                "bidirectional_control_mean": handshake_bidirectional,
                "coordinator_delegated_path_accesses": 0,
            },
        },
    }
    (path / "ownership-selection.json").write_text(json.dumps(ownership))
    (path / "natural-selection.json").write_text(json.dumps(natural))


def test_classifies_non_monotonic_consolidation(tmp_path: Path) -> None:
    _write_step(tmp_path, 1, gains=2, followup_causal=2)
    _write_step(tmp_path, 2, gains=4, followup_causal=4)

    report = summarize(tmp_path)

    assert report["classification"] == "CONSOLIDATION_EVIDENCE"
    assert report["first_consolidation_update"] == 2


def test_hard_invariant_failure_overrides_partial_gain(tmp_path: Path) -> None:
    _write_step(tmp_path, 1, gains=5, followup_causal=4, child_path=1)

    report = summarize(tmp_path)

    assert report["classification"] == "BRANCH_REJECTED_HARD_INVARIANT"
    assert report["first_hard_failure_update"] == 1


def test_does_not_call_tradeoff_stable_before_frozen_horizon(tmp_path: Path) -> None:
    _write_step(tmp_path, 1, gains=4, followup_causal=3)

    report = summarize(tmp_path)

    assert report["classification"] == "CONTINUATION_INCONCLUSIVE"
    assert report["trajectory_complete"] is False
