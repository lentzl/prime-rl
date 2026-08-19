"""Audit whether natural-yield feedback creates a usable SDPO teacher signal."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from scripts import validate_natural_yield_sdpo_zero_lr_v1 as zero
from scripts.validate_prime_agent_sdpo_zero_lr_audit_v1 import _read_json, _read_jsonl


class DistributionAuditFailure(ValueError):
    """The run does not expose a complete, trustworthy teacher distribution."""


EXPECTED_STATE_COUNT = 8
LOG_ODDS_EPSILON = 1e-12


def _bounded_log_odds(probability: float) -> float:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise DistributionAuditFailure(f"invalid probability for log odds: {probability}")
    bounded = min(max(probability, LOG_ODDS_EPSILON), 1 - LOG_ODDS_EPSILON)
    return math.log(bounded) - math.log1p(-bounded)


def _find_subsequence(values: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    for start in range(len(values) - len(needle) + 1):
        if values[start : start + len(needle)] == needle:
            return start
    return None


def _probability(support: dict[str, Any], token_id: int, source: str) -> float | None:
    token_ids = support.get("token_ids")
    logprobs = support.get(f"{source}_logprobs")
    if not isinstance(token_ids, list) or not isinstance(logprobs, list) or len(token_ids) != len(logprobs):
        raise DistributionAuditFailure(f"malformed {source} support distribution")
    try:
        index = token_ids.index(token_id)
    except ValueError:
        return None
    value = logprobs[index]
    if not isinstance(value, int | float) or not math.isfinite(value) or value > 1e-6:
        raise DistributionAuditFailure(f"invalid {source} logprob for token {token_id}")
    return math.exp(value)


def _validate_support(entry: dict[str, Any], topk: int) -> None:
    for name in ("student_support", "teacher_support"):
        support = entry.get(name)
        if not isinstance(support, dict):
            raise DistributionAuditFailure(f"active token is missing {name}")
        token_ids = support.get("token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != topk or len(set(token_ids)) != topk:
            raise DistributionAuditFailure(f"{name} does not contain {topk} distinct token ids")
        for source in ("student", "teacher"):
            _probability(support, token_ids[0], source)

    teacher = entry["teacher_support"]
    teacher_logprobs = teacher["teacher_logprobs"]
    if teacher_logprobs != sorted(teacher_logprobs, reverse=True):
        raise DistributionAuditFailure("teacher-owned support is not sorted by teacher probability")


def _token_label(tokenizer: Any, token_id: int) -> dict[str, Any]:
    return {
        "id": token_id,
        "token": tokenizer.convert_ids_to_tokens(token_id),
        "text": tokenizer.decode([token_id], skip_special_tokens=False),
    }


def _decision_record(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    topk: int,
    tool_marker_ids: list[int],
    yield_token_id: int,
) -> dict[str, Any]:
    supports = record.get("sdpo_support")
    replays = record.get("sdpo_teacher_replays")
    if not isinstance(supports, list) or not isinstance(replays, list) or len(replays) != 1:
        raise DistributionAuditFailure("each active sample must expose one teacher replay")
    by_position = {entry.get("position"): entry for entry in supports if isinstance(entry, dict)}
    active_positions = [
        index
        for index, (masked, weight) in enumerate(
            zip(record.get("loss_mask", []), record.get("sdpo_weights", []), strict=True)
        )
        if masked and weight is not None and weight != 0
    ]
    if sorted(by_position) != active_positions:
        raise DistributionAuditFailure("sparse support positions do not exactly match active SDPO tokens")
    for entry in supports:
        _validate_support(entry, topk)

    replay = replays[0]
    completion_ids = replay.get("completion_ids")
    target_offsets = replay.get("target_offsets")
    student_positions = replay.get("student_positions")
    if not all(isinstance(values, list) for values in (completion_ids, target_offsets, student_positions)):
        raise DistributionAuditFailure("teacher replay token routing is malformed")
    if len(target_offsets) != len(student_positions) or len(target_offsets) != len(active_positions):
        raise DistributionAuditFailure("teacher replay target routing disagrees with sparse support")
    marker_offset = _find_subsequence(completion_ids, tool_marker_ids)
    if marker_offset is None:
        raise DistributionAuditFailure("failed response does not contain the tool-call marker")
    offset_to_position = dict(zip(target_offsets, student_positions, strict=True))
    marker_position = offset_to_position.get(marker_offset)
    if marker_position is None or marker_position not in by_position:
        raise DistributionAuditFailure("tool-call decision token is outside the active SDPO target")

    entry = by_position[marker_position]
    student_support = entry["student_support"]
    teacher_support = entry["teacher_support"]
    marker_token_id = tool_marker_ids[0]
    student_tool_probability = _probability(student_support, marker_token_id, "student")
    teacher_tool_probability = _probability(student_support, marker_token_id, "teacher")
    if student_tool_probability is None or teacher_tool_probability is None:
        raise DistributionAuditFailure("student support does not expose the failed tool-call decision")

    student_top_id = student_support["token_ids"][0]
    teacher_top_id = teacher_support["token_ids"][0]
    teacher_top_in_student_support = teacher_top_id in student_support["token_ids"]
    overlap = len(set(student_support["token_ids"]) & set(teacher_support["token_ids"]))
    return {
        "export_sequence_idx": record.get("export_sequence_idx"),
        "decision_position": marker_position,
        "student_top": _token_label(tokenizer, student_top_id),
        "teacher_top": _token_label(tokenizer, teacher_top_id),
        "teacher_top_is_tool": teacher_top_id == marker_token_id,
        "teacher_top_in_student_support": teacher_top_in_student_support,
        "support_overlap": overlap,
        "student_tool_probability": student_tool_probability,
        "teacher_tool_probability": teacher_tool_probability,
        "teacher_to_student_tool_ratio": teacher_tool_probability / student_tool_probability,
        "student_yield_probability": _probability(student_support, yield_token_id, "student"),
        "teacher_yield_probability_on_student_support": _probability(
            student_support, yield_token_id, "teacher"
        ),
        "teacher_yield_probability": _probability(teacher_support, yield_token_id, "teacher"),
        "yield_token_in_student_support": yield_token_id in student_support["token_ids"],
        "yield_token_in_teacher_support": yield_token_id in teacher_support["token_ids"],
    }


def _distribution_analysis(
    run_dir: Path,
    model_path: str,
    *,
    expected_state_count: int = EXPECTED_STATE_COUNT,
) -> dict[str, Any]:
    trainer = _read_json(run_dir / "configs" / "trainer.json")
    if trainer.get("enable_sdpo_support_export") is not True:
        raise DistributionAuditFailure("resolved trainer did not enable SDPO support export")
    topk = trainer.get("sdpo_loss", {}).get("distillation_topk")
    if not isinstance(topk, int) or topk <= 0:
        raise DistributionAuditFailure("resolved trainer has no positive SDPO top-k")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tool_marker_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    yield_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not tool_marker_ids or not isinstance(yield_token_id, int) or yield_token_id < 0:
        raise DistributionAuditFailure("tokenizer does not expose Prime Agent control tokens")

    export_dir = run_dir / "token_exports" / "step_1"
    records = [
        record
        for path in sorted(export_dir.glob("rank_*.jsonl"))
        for record in _read_jsonl(path)
        if any(value is not None and value != 0 for value in record.get("sdpo_weights", []))
    ]
    decisions = [
        _decision_record(
            record,
            tokenizer=tokenizer,
            topk=topk,
            tool_marker_ids=tool_marker_ids,
            yield_token_id=yield_token_id,
        )
        for record in records
    ]
    if len(decisions) != expected_state_count:
        raise DistributionAuditFailure(
            f"expected {expected_state_count} active coordinator decisions, found {len(decisions)}"
        )

    ratios = [decision["teacher_to_student_tool_ratio"] for decision in decisions]
    log_odds_shifts = [
        _bounded_log_odds(decision["student_tool_probability"])
        - _bounded_log_odds(decision["teacher_tool_probability"])
        for decision in decisions
    ]
    non_tool_argmax_rate = sum(not decision["teacher_top_is_tool"] for decision in decisions) / len(decisions)
    teacher_top_coverage = sum(decision["teacher_top_in_student_support"] for decision in decisions) / len(decisions)
    median_tool_ratio = statistics.median(ratios)
    positive_shift_rate = sum(shift > 0 for shift in log_odds_shifts) / len(log_odds_shifts)
    aggregate_shift = statistics.fmean(log_odds_shifts)
    teacher_signal_present = positive_shift_rate >= 0.75 and aggregate_shift > 0
    current_support_connected = teacher_top_coverage == 1.0
    if not teacher_signal_present:
        next_action = "change_feedback_or_teacher_source"
    elif not current_support_connected:
        next_action = "expand_sdpo_support_before_training"
    else:
        next_action = "sample_exact_conditioned_teacher_before_training"
    return {
        "topk": topk,
        "tool_marker": [_token_label(tokenizer, token_id) for token_id in tool_marker_ids],
        "yield_token": _token_label(tokenizer, yield_token_id),
        "decisions": decisions,
        "summary": {
            "decision_count": len(decisions),
            "teacher_non_tool_argmax_rate": non_tool_argmax_rate,
            "median_teacher_to_student_tool_probability_ratio": median_tool_ratio,
            "teacher_away_from_tool_log_odds_shifts": log_odds_shifts,
            "positive_away_from_tool_shift_rate": positive_shift_rate,
            "mean_away_from_tool_log_odds_shift": aggregate_shift,
            "teacher_top1_covered_by_student_support_rate": teacher_top_coverage,
            "mean_support_overlap": statistics.fmean(decision["support_overlap"] for decision in decisions),
            "teacher_signal_present": teacher_signal_present,
            "current_student_selected_support_connected": current_support_connected,
            "next_action": next_action,
        },
    }


def _validate_state_diversity(run_dir: Path, expected_state_count: int) -> dict[str, Any]:
    records = _read_jsonl(
        run_dir / "rollouts" / "step_1" / "train" / "effective" / "traces.jsonl"
    )
    episode_ids = []
    semantic_families = []
    phrasing_variants = []
    for index, record in enumerate(records):
        task_data = record.get("task", {}).get("data", {})
        episode_id = task_data.get("episode_id")
        metadata = task_data.get("generation_metadata", {})
        semantic_family = metadata.get("semantic_family")
        phrasing = metadata.get("control_contract_variant")
        if not all(isinstance(value, str) and value for value in (episode_id, semantic_family, phrasing)):
            raise DistributionAuditFailure(f"effective state {index} lacks generation identity")
        episode_ids.append(episode_id)
        semantic_families.append(semantic_family)
        phrasing_variants.append(phrasing)
    if len(set(episode_ids)) != expected_state_count:
        raise DistributionAuditFailure("teacher audit states are not all distinct")
    if len(set(semantic_families)) < 2:
        raise DistributionAuditFailure("teacher audit does not span multiple semantic families")
    if len(set(phrasing_variants)) < 2:
        raise DistributionAuditFailure("teacher audit does not span multiple phrasing variants")
    return {
        "distinct_states": len(set(episode_ids)),
        "semantic_families": sorted(set(semantic_families)),
        "phrasing_variants": sorted(set(phrasing_variants)),
    }


def validate(
    run_dir: Path,
    expected_model_path: str,
    *,
    expected_state_count: int = EXPECTED_STATE_COUNT,
) -> dict[str, Any]:
    zero._validate_configs(
        run_dir,
        expected_model_path,
        expected_batch_size=expected_state_count,
    )
    traces = zero._validate_traces(run_dir, expected_batch_size=expected_state_count)
    routing = zero._validate_token_routing(
        run_dir,
        traces,
        expected_batch_size=expected_state_count,
    )
    metrics = zero._validate_metrics(run_dir)
    zero._validate_no_model_artifacts(run_dir)
    diversity = _validate_state_diversity(run_dir, expected_state_count)
    distribution = _distribution_analysis(
        run_dir,
        expected_model_path,
        expected_state_count=expected_state_count,
    )
    return {
        "verdict": "pass",
        "mechanism": "natural-yield-feedback-conditioned-teacher-distribution-audit",
        "expected_model_path": expected_model_path,
        "model_artifacts_written": False,
        "metrics": metrics,
        "token_routing": routing,
        "state_diversity": diversity,
        "distribution": distribution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-model-path", required=True)
    parser.add_argument("--expected-state-count", type=int, default=EXPECTED_STATE_COUNT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_state_count < EXPECTED_STATE_COUNT:
        parser.error(f"expected-state-count must be at least {EXPECTED_STATE_COUNT}")
    report = validate(
        args.run_dir,
        args.expected_model_path,
        expected_state_count=args.expected_state_count,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
