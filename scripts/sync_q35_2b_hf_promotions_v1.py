#!/usr/bin/env python3
"""Reconcile admitted role-policy frontiers to latest-only HF repositories."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import publish_q35_2b_hf_latest_remote_v1 as publisher
import run_q35_2b_role_grpo_autonomous_v1 as controller

PUBLICATION_SCHEMA = "qwen35-2b-hf-promotion-publication-event/v1"
ROLES = ("coordinator", "child")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _digest(event: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(event).encode()).hexdigest()


def load_publications(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    previous = None
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        event = json.loads(line)
        digest = event.pop("event_sha256", None)
        if (
            event.get("schema_version") != PUBLICATION_SCHEMA
            or event.get("sequence") != len(events)
            or event.get("previous_event_sha256") != previous
            or digest != _digest(event)
        ):
            raise ValueError(f"invalid publication event chain at {path}:{line_number}")
        event["event_sha256"] = digest
        events.append(event)
        previous = digest
    return events


def append_publication(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        existing = [json.loads(line) for line in handle if line.strip()]
        event = {
            "schema_version": PUBLICATION_SCHEMA,
            "sequence": len(existing),
            "previous_event_sha256": existing[-1]["event_sha256"] if existing else None,
            "recorded_at_utc": _now(),
            **payload,
        }
        event["event_sha256"] = _digest(event)
        handle.seek(0, 2)
        handle.write(_canonical(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        return event


def _token(path: Path) -> str:
    payload = json.loads(path.read_text())
    token = payload.get("token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        raise ValueError(f"HF secret requires a non-empty token: {path}")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(f"HF secret must not be group/world accessible: {path}")
    return token


def _promotion_evidence(events: list[dict[str, Any]], role: str, candidate: dict[str, str]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("kind") == "evaluation_completed" and event.get("role") == role and event.get("admitted") is True:
            cycle = event["cycle"]
            trained = next(
                (
                    item
                    for item in events
                    if item.get("kind") == "train_completed" and item.get("cycle") == cycle and item.get("role") == role
                ),
                None,
            )
            if trained and trained["output_candidate"]["model_sha256"] == candidate["model_sha256"]:
                return {
                    "source_kind": "evaluation_completed",
                    "cycle": cycle,
                    "qualifying": event.get("qualifying"),
                    "distinct_qualifying": event.get("distinct_qualifying"),
                    "promotion_minimum": controller.PROMOTION_MINIMUM,
                    "run_dir": event.get("run_dir"),
                }
    initial = events[0]
    initial_candidate = initial.get("promoted", {}).get(role)
    if initial_candidate == candidate:
        admission = initial.get("initial_admission", {})
        return {
            "source_kind": "initialized",
            "cycle": None,
            "qualifying": admission.get("qualifying"),
            "distinct_qualifying": admission.get("distinct_qualifying"),
            "promotion_minimum": controller.PROMOTION_MINIMUM,
            "run_dir": admission.get("run_dir"),
        }
    raise ValueError(f"cannot locate admission evidence for promoted {role} candidate")


def _model_card(*, role: str, candidate: dict[str, str], evidence: dict[str, Any]) -> str:
    return f"""---
library_name: transformers
pipeline_tag: text-generation
---

# Qwen3.5-2B Prime Agent {role.title()} Policy

This repository is an automatically maintained **latest-only** publication of the
currently admitted `{role}` policy in the Prime Agent role-GRPO experiment.

- Label: `{candidate["label"]}`
- Dense-weight SHA-256: `{candidate["model_sha256"]}`
- Admission source: `{evidence["source_kind"]}`
- Distinct qualifying trajectories: `{evidence["distinct_qualifying"]}`
- Required qualifying trajectories: `{evidence["promotion_minimum"]}`

Older checkpoints are deliberately removed when a newer admitted checkpoint is
published. Publication is downstream of admission and never triggers training.
"""


def _completed_keys(events: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(event["role"], event["model_sha256"]) for event in events if event.get("kind") == "publication_completed"}


def reconcile_once(args: argparse.Namespace) -> int:
    autonomous_events = controller.load_events(args.events_file)
    if not autonomous_events:
        return 0
    state = controller.project(autonomous_events)
    publications = load_publications(args.publication_events)
    completed = _completed_keys(publications)
    token = _token(args.secret_file)
    published = 0
    for role in ROLES:
        candidate = state["promoted"][role]
        key = (role, candidate["model_sha256"])
        reconciliation: dict[str, Any] = {}
        if key in completed:
            try:
                remote_sha = publisher.remote_checkpoint_sha256(
                    token=token,
                    repo_id=getattr(args, f"{role}_repo"),
                )
            except Exception:
                remote_sha = None
            if remote_sha == candidate["model_sha256"]:
                continue
            reconciliation = {
                "reconciliation_reason": "remote_head_missing_or_mismatched",
                "observed_remote_model_sha256": remote_sha,
            }
        checkpoint = Path(candidate["model_path"])
        actual_sha = controller._sha256(checkpoint / "model.safetensors")
        if actual_sha != candidate["model_sha256"]:
            raise ValueError(f"promoted {role} checkpoint hash mismatch: {checkpoint}")
        evidence = _promotion_evidence(autonomous_events, role, candidate)
        repo_id = getattr(args, f"{role}_repo")
        append_publication(
            args.publication_events,
            {
                "kind": "publication_started",
                "role": role,
                "repo_id": repo_id,
                "label": candidate["label"],
                "model_path": candidate["model_path"],
                "model_sha256": candidate["model_sha256"],
                "admission": evidence,
                **reconciliation,
            },
        )
        try:
            revision = publisher.publish_checkpoint(
                token=token,
                repo_id=repo_id,
                checkpoint_dir=checkpoint,
                model_card=_model_card(role=role, candidate=candidate, evidence=evidence),
                commit_message=(f"Publish admitted {role} {candidate['label']} ({candidate['model_sha256'][:12]})"),
            )
            published_sha = publisher.remote_checkpoint_sha256(token=token, repo_id=repo_id)
            if published_sha != candidate["model_sha256"]:
                raise RuntimeError(
                    f"published {role} head hash mismatch: expected {candidate['model_sha256']}, "
                    f"observed {published_sha}"
                )
        except Exception as error:
            message = str(error).replace(token, "[redacted]")[:1000]
            append_publication(
                args.publication_events,
                {
                    "kind": "publication_failed",
                    "role": role,
                    "repo_id": repo_id,
                    "label": candidate["label"],
                    "model_sha256": candidate["model_sha256"],
                    "error_type": type(error).__name__,
                    "error": message,
                },
            )
            continue
        append_publication(
            args.publication_events,
            {
                "kind": "publication_completed",
                "role": role,
                "repo_id": repo_id,
                "label": candidate["label"],
                "model_sha256": candidate["model_sha256"],
                "revision": revision,
            },
        )
        completed.add(key)
        published += 1
    return published


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-file", type=Path, required=True)
    parser.add_argument("--publication-events", type=Path, required=True)
    parser.add_argument("--secret-file", type=Path, required=True)
    parser.add_argument("--coordinator-repo", required=True)
    parser.add_argument("--child-repo", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    lock_path = args.publication_events.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another HF promotion reconciler holds the lock") from error
        while True:
            reconcile_once(args)
            if args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
