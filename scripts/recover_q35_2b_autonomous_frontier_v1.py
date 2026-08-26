#!/usr/bin/env python3
"""Fork a validated autonomous event prefix onto an explicitly recovered frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import run_q35_2b_role_grpo_autonomous_v1 as controller


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recover(args: argparse.Namespace) -> dict:
    source_events_path = args.source_state_dir.resolve() / "events.jsonl"
    target_state_dir = args.target_state_dir.resolve()
    target_events_path = target_state_dir / "events.jsonl"
    if target_events_path.exists():
        raise FileExistsError(f"refusing to overwrite recovery ledger: {target_events_path}")

    source_events = controller.load_events(source_events_path)
    if args.through_sequence >= len(source_events):
        raise ValueError("recovery prefix exceeds the source event ledger")
    prefix = source_events[: args.through_sequence + 1]
    terminal = prefix[-1]
    if terminal.get("kind") != "evaluation_completed":
        raise ValueError("recovery prefix must end at evaluation_completed")

    coordinator = controller._candidate(
        args.coordinator_model.resolve(), args.coordinator_label
    )
    child = controller._candidate(args.child_model.resolve(), args.child_label)
    if coordinator["model_sha256"] != args.coordinator_sha256:
        raise ValueError("coordinator recovery checkpoint hash mismatch")
    if child["model_sha256"] != args.child_sha256:
        raise ValueError("child recovery checkpoint hash mismatch")

    target_state_dir.mkdir(parents=True, exist_ok=False)
    with target_events_path.open("x", encoding="utf-8") as stream:
        for event in prefix:
            stream.write(controller._canonical(event) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    controller.load_events(target_events_path)

    return controller.append_event(
        target_events_path,
        {
            "kind": "frontier_recovered",
            "frontier": {"coordinator": coordinator, "child": child},
            "next_role": args.next_role,
            "next_cycle": args.next_cycle,
            "recovery_reason": args.recovery_reason,
            "source_state_dir": str(args.source_state_dir.resolve()),
            "source_event_count": len(source_events),
            "source_events_sha256": _file_sha256(source_events_path),
            "source_terminal_sequence": args.through_sequence,
            "discarded_cycle": args.discarded_cycle,
            "discarded_model_sha256": args.discarded_model_sha256,
            "discarded_evaluation_run": args.discarded_evaluation_run,
            "duplicate_evaluation_launched": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state-dir", type=Path, required=True)
    parser.add_argument("--target-state-dir", type=Path, required=True)
    parser.add_argument("--through-sequence", type=int, required=True)
    parser.add_argument("--coordinator-model", type=Path, required=True)
    parser.add_argument("--coordinator-label", required=True)
    parser.add_argument("--coordinator-sha256", required=True)
    parser.add_argument("--child-model", type=Path, required=True)
    parser.add_argument("--child-label", required=True)
    parser.add_argument("--child-sha256", required=True)
    parser.add_argument("--next-cycle", type=int, required=True)
    parser.add_argument("--next-role", choices=("coordinator", "child"), required=True)
    parser.add_argument("--recovery-reason", required=True)
    parser.add_argument("--discarded-cycle", type=int, required=True)
    parser.add_argument("--discarded-model-sha256", required=True)
    parser.add_argument("--discarded-evaluation-run", required=True)
    args = parser.parse_args()
    if min(args.through_sequence, args.next_cycle, args.discarded_cycle) < 0:
        raise ValueError("recovery sequence and cycle values must be nonnegative")
    print(json.dumps(recover(args), sort_keys=True))


if __name__ == "__main__":
    main()
