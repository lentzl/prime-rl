#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from prime_rl.trainer.rl.sdpo_export_verify import verify_sdpo_token_exports


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify schema-v2 SDPO token export artifacts.")
    parser.add_argument(
        "path",
        type=Path,
        help="Output directory, token_exports directory, step directory, or rank_*.jsonl file.",
    )
    parser.add_argument(
        "--require-stable",
        action="store_true",
        help="Require each token export step directory to contain a STABLE marker.",
    )
    parser.add_argument(
        "--require-student-preflight",
        action="store_true",
        help=(
            "Require preflight-only student support rows and final transported support rows to overlap "
            "on the same training step, sample_id, env-aware sample signature, and weighted token rows, "
            "with final teacher logprobs distinct from trainer-forward student logprobs."
        ),
    )
    parser.add_argument(
        "--require-importance-ratio-evidence",
        action="store_true",
        help=(
            "Require final weighted SDPO rows to carry rollout-IS ratio evidence "
            "(log_importance_ratio, importance_ratio, prob_delta)."
        ),
    )
    parser.add_argument(
        "--expected-topk",
        type=int,
        default=None,
        help=(
            "Require every supported SDPO top-k row to have this width. Final SDPO records must "
            "also carry transported teacher support at every weighted token row."
        ),
    )
    parser.add_argument(
        "--rollout-is-threshold",
        type=float,
        default=None,
        help=(
            "Require every final weighted SDPO row to carry sdpo_rollout_is_weights at most this "
            "rollout-IS truncation threshold."
        ),
    )
    parser.add_argument(
        "--rollout-is",
        choices=("token", "sequence"),
        default=None,
        help=(
            "When set with --rollout-is-threshold, also require sdpo_rollout_is_weights to match "
            "the selected truncated rollout-IS mode."
        ),
    )
    args = parser.parse_args()

    stats = verify_sdpo_token_exports(
        args.path,
        require_stable=args.require_stable,
        require_student_preflight=args.require_student_preflight,
        require_importance_ratio_evidence=args.require_importance_ratio_evidence,
        expected_topk=args.expected_topk,
        rollout_is_threshold=args.rollout_is_threshold,
        rollout_is=args.rollout_is,
    )
    print(
        "Verified SDPO token exports: "
        f"files={stats.files}, records={stats.records}, sdpo_records={stats.sdpo_records}, "
        f"transported_rows={stats.transported_rows}, student_rows={stats.student_rows}, "
        f"paired_rows={stats.paired_rows}, matching_support_rows={stats.matching_support_rows}, "
        f"distinct_teacher_logprob_rows={stats.distinct_teacher_logprob_rows}, "
        f"importance_ratio_rows={stats.importance_ratio_rows}, "
        f"rollout_is_weight_rows={stats.rollout_is_weight_rows}, "
        f"student_preflight_rows={stats.student_preflight_rows}, temperature_rows={stats.temperature_rows}, "
        f"sample_id_records={stats.sample_id_records}, "
        f"stable_steps={stats.stable_steps}, "
        f"paired_steps={list(stats.paired_step_names)}, "
        f"matching_support_steps={list(stats.matching_support_step_names)}, "
        f"student_preflight_steps={list(stats.student_preflight_step_names)}, "
        f"matched_support_samples={len(stats.matched_support_sample_keys)}, "
        f"matched_support_token_rows={len(stats.matched_support_row_keys)}, "
        f"distinct_teacher_logprob_token_rows={len(stats.distinct_teacher_logprob_row_keys)}, "
        f"importance_ratio_token_rows={len(stats.importance_ratio_row_keys)}, "
        f"rollout_is_weight_token_rows={len(stats.rollout_is_weight_row_keys)}"
    )


if __name__ == "__main__":
    main()
