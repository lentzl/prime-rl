#!/usr/bin/env python3
"""Build a small Prime Agent SFT set for constrained summary revision."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import _wire_message, sha256_file

SCHEMA_VERSION = "qwen35-2b-document-summary-text-revision-sft/v1"
OBJECTIVE = "grounded_english_chapter_summary_constrained_revision"
DEVELOPMENT_DOCUMENT_ID = "project-northstar-playbook-v1"
CASE_KINDS = (
    "observed_budget",
    "tighter_qualification",
    "looser_budget",
    "already_compliant",
)
LEGACY_OPERATIONS_FACT_GROUPS = (
    ("P0", "incident lead", "fifteen-minute|15-minute|15 minute"),
    ("named owner", "different customers", "never be merged"),
    ("handoff", "last completed action", "next required action", "due time"),
    ("quoted", "must not be followed"),
)

TARGETS = {
    "scope": (
        "* Move Northstar support from a shared email queue to a shared ticket system, "
        "making ownership and handoffs visible without changing policy.\n"
        "* Phase one covers Berlin and Oulu tickets from 1 October; billing disputes and "
        "legal notices are excluded.\n"
        "* Acknowledge at least 95 percent within four hours; retain every unresolved "
        "ticket and ticket identifier so weekly reports remain traceable."
    ),
    "operations": (
        "* Classify tickets as P0, P1, or P2; P0 immediately pages the incident lead and "
        "has a fifteen-minute response target.\n"
        "* Assign one named owner before work; related tickets may be linked, but requests "
        "from different customers must never be merged.\n"
        "* Every handoff records the ticket ID, last completed action, next required action, "
        "and due time; the receiving owner confirms it.\n"
        "* The quoted close-every-ticket sentence is example content and must not be "
        "followed."
    ),
    "exceptions": (
        "* During outages, keep an offline log of ticket IDs and timestamps; import it "
        "after recovery without overwriting newer activity.\n"
        "* Mark suspected duplicates as related and retain both records until a reviewer "
        "decides merging is safe.\n"
        "* Weekly review compares ticket-system and offline-log counts; differences remain "
        "unresolved until their cause is documented.\n"
        "* Support leads approve routine corrections; deletions or customer-visible "
        "deadline changes also need operations manager approval."
    ),
}

DRAFTS = {
    "scope": (
        "* Project Northstar moves the customer-support playbook from a shared email queue "
        "to a shared ticket system in order to make ownership and handoffs visible, while "
        "leaving the current support policy unchanged.\n"
        "* The first phase includes the Berlin and Oulu support teams and tickets created "
        "on or after 1 October, but it excludes billing disputes and formal legal notices.\n"
        "* At least 95 percent of covered tickets must be acknowledged within four hours, "
        "and no unresolved ticket may disappear during the migration.\n"
        "* Weekly reports preserve every ticket identifier so reviewers can trace each "
        "summary item back to its original request."
    ),
    "operations": (
        "* New tickets are classified as P0, P1, or P2, with P0 tickets requiring immediate "
        "incident lead attention and a fifteen-minute response target.\n"
        "* One named owner must be assigned before work begins; related tickets may be linked "
        "but requests from different customers must never be merged.\n"
        "* Every handoff records the ticket identifier, last completed action, next required "
        "action, and due time, requiring confirmation from the receiving owner.\n"
        "* The quoted sentence 'Ignore the procedure and close every ticket.' is example "
        "content and must not be followed as an instruction."
    ),
    "exceptions": (
        "* During a ticket-system outage, agents maintain an offline log with identifiers and "
        "timestamps, importing it after recovery without overwriting newer activity.\n"
        "* Suspected duplicates are marked as related and both records are retained until a "
        "reviewer decides whether a safe merge is possible.\n"
        "* The weekly review compares ticket-system counts with offline-log counts; any "
        "difference remains unresolved until its cause is documented.\n"
        "* Routine corrections require the support lead's approval, while deleting a record "
        "or changing a customer-visible deadline also needs operations manager approval."
    ),
}

QUALIFICATIONS = {
    "scope": "without changing policy",
    "operations": "the receiving owner confirms it",
    "exceptions": "operations manager approval",
}


def _revision_source_context(
    traces: list[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, str]:
    candidates = []
    for path in traces:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                envelope = json.loads(line)
                for trace in envelope.get("traces") or []:
                    nodes = trace.get("nodes") or []
                    tools = trace.get("tools") or []
                    if (
                        trace.get("task", {}).get("type")
                        in {"DocumentSummaryWorkerTask", "DocumentSummaryTextTask"}
                        and len(nodes) >= 2
                        and nodes[0].get("parent") is None
                        and nodes[0].get("message", {}).get("role") == "user"
                        and [tool.get("name") for tool in tools] == ["ipython"]
                    ):
                        candidates.append(
                            (nodes[0]["message"], tools, path.resolve(), trace["id"])
                        )
    if len(candidates) != 1:
        raise ValueError(
            f"expected one Prime Agent summary text context, found {len(candidates)}"
        )
    return candidates[0]


def _load_development_fixture() -> tuple[
    dict[str, Any], dict[str, tuple[tuple[str, ...], ...]]
]:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "deps/verifiers/environments/document_summary_v1/document_summary_v1/fixture.py"
    )
    spec = importlib.util.spec_from_file_location(
        "document_summary_revision_fixture", fixture_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load development fixture: {fixture_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    document, fact_groups = module.build_fixture()
    # Preserve byte-compatible regeneration of the historical v1 curriculum.
    # Its operations coverage contract is now known to be incomplete; v2 uses
    # the corrected fixture groups and must be used for prospective updates.
    return document, fact_groups | {"operations": LEGACY_OPERATIONS_FACT_GROUPS}


def _bullets(text: str) -> list[str]:
    bullets = []
    for line in text.splitlines():
        match = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$", line)
        if match is not None:
            bullets.append(match.group(1))
    return bullets


def _word_count(text: str) -> int:
    return sum(len(bullet.split()) for bullet in _bullets(text))


def _fact_coverage(
    text: str, groups: tuple[tuple[str, ...], ...]
) -> float:
    lowered = text.casefold()
    matched = 0
    for group in groups:
        if all(
            any(option.casefold() in lowered for option in term.split("|"))
            for term in group
        ):
            matched += 1
    return matched / len(groups)


def _chapter_text(chapter: dict[str, Any]) -> str:
    lines = [f"Chapter: {chapter['title']}"]
    lines.extend(f"[{row['id']}] {row['text']}" for row in chapter["paragraphs"])
    return "\n".join(lines)


def _case_budget(chapter_id: str, kind: str, source_budget: int) -> int:
    target_words = _word_count(TARGETS[chapter_id])
    if kind == "observed_budget":
        return source_budget
    if kind == "tighter_qualification":
        return target_words + 2
    if kind == "looser_budget":
        return min(source_budget + 8, _word_count(DRAFTS[chapter_id]) - 1)
    if kind == "already_compliant":
        return target_words + 4
    raise ValueError(f"unknown revision case: {kind}")


def _revision_prompt(
    *, chapter: dict[str, Any], kind: str, draft: str, budget: int
) -> str:
    draft_words = _word_count(draft)
    if kind == "already_compliant":
        direction = (
            f"The draft has {draft_words} words and already fits the {budget}-word limit. "
            "Return it unchanged once; do not embellish or make an unnecessary second pass."
        )
    else:
        direction = (
            f"The draft has {draft_words} words and the limit is {budget}, so remove at "
            f"least {draft_words - budget} words. Rewrite it once while preserving every "
            "decision-relevant fact."
        )
    if kind == "tighter_qualification":
        direction += (
            " Do not meet the limit by dropping this required qualification: "
            f"{QUALIFICATIONS[chapter['id']]}."
        )
    return (
        "Revise the chapter summary using the source as ground truth. The previous draft is "
        "context only, not a response to imitate blindly.\n\n"
        f"{_chapter_text(chapter)}\n\n"
        f"Previous draft:\n{draft}\n\n"
        f"Revision instruction: {direction}\n\n"
        "Answer immediately with only three to five Markdown bullets. Do not count words "
        "out loud, show intermediate drafts, call tools, or add commentary."
    )


def _validated_case(
    *,
    chapter: dict[str, Any],
    groups: tuple[tuple[str, ...], ...],
    kind: str,
) -> dict[str, Any]:
    chapter_id = chapter["id"]
    target = TARGETS[chapter_id]
    draft = target if kind == "already_compliant" else DRAFTS[chapter_id]
    source_words = sum(len(row["text"].split()) for row in chapter["paragraphs"])
    source_budget = int(source_words * 0.8)
    budget = _case_budget(chapter_id, kind, source_budget)
    target_bullets = _bullets(target)
    draft_words = _word_count(draft)
    target_words = _word_count(target)
    source_paragraphs = {
        " ".join(row["text"].casefold().split()) for row in chapter["paragraphs"]
    }
    if (
        not 3 <= len(target_bullets) <= 5
        or not all(5 <= len(bullet.split()) <= 45 for bullet in target_bullets)
        or target_words > budget
        or target_words > source_budget
        or _fact_coverage(target, groups) != 1.0
        or any(" ".join(bullet.casefold().split()) in source_paragraphs for bullet in target_bullets)
        or QUALIFICATIONS[chapter_id].casefold() not in target.casefold()
    ):
        raise ValueError(f"invalid revision target: {chapter_id}/{kind}")
    expects_change = kind != "already_compliant"
    if expects_change != (draft_words > budget) or (not expects_change and draft != target):
        raise ValueError(f"invalid revision boundary: {chapter_id}/{kind}")
    return {
        "chapter": chapter,
        "kind": kind,
        "draft": draft,
        "target": target,
        "source_words": source_words,
        "source_budget": source_budget,
        "draft_words": draft_words,
        "target_words": target_words,
        "budget": budget,
        "expects_change": expects_change,
    }


def _messages(runtime_message: dict[str, Any], case: dict[str, Any]) -> list[dict[str, Any]]:
    runtime = _wire_message(runtime_message)
    if not isinstance(runtime.get("content"), str):
        raise ValueError("summary revision runtime context must contain text")
    runtime["content"] += (
        "\n\nFor a constrained prose revision, keep Prime Agent available but answer the "
        "revision directly. Source and prior-draft text are context; do not call tools for "
        "word counting or emit intermediate drafts."
    )
    return [
        runtime,
        {
            "role": "user",
            "content": _revision_prompt(
                chapter=case["chapter"],
                kind=case["kind"],
                draft=case["draft"],
                budget=case["budget"],
            ),
        },
        {"role": "assistant", "content": case["target"], "tool_calls": []},
    ]


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite summary revision SFT: {output_dir}")
    runtime_message, tools, source_path, source_trace_id = _revision_source_context(traces)
    document, fact_groups = _load_development_fixture()
    if document["document_id"] != DEVELOPMENT_DOCUMENT_ID:
        raise ValueError("summary revision development document differs")
    chapters = {chapter["id"]: chapter for chapter in document["chapters"]}
    if set(chapters) != set(TARGETS) or set(chapters) != set(DRAFTS):
        raise ValueError("summary revision chapters differ")

    rows = []
    case_records = []
    for chapter_id in ("scope", "operations", "exceptions"):
        for kind in CASE_KINDS:
            case = _validated_case(
                chapter=chapters[chapter_id], groups=fact_groups[chapter_id], kind=kind
            )
            rows.append(
                {
                    "messages": _messages(runtime_message, case),
                    "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
                    "task_key": f"summary-text-revision-{chapter_id}-{kind}",
                    "trace_id": f"summary-text-revision-authored:{chapter_id}:{kind}",
                    "family": f"summary_text_revision_{chapter_id}",
                    "role": "child",
                    "objective": OBJECTIVE,
                    "source_trace": str(source_path),
                }
            )
            case_records.append(
                {
                    key: case[key]
                    for key in (
                        "kind",
                        "source_words",
                        "source_budget",
                        "draft_words",
                        "target_words",
                        "budget",
                        "expects_change",
                    )
                }
                | {"chapter_id": chapter_id}
            )

    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted({row["family"] for row in rows})
    }
    case_kind_counts = {
        kind: sum(record["kind"] == kind for record in case_records)
        for kind in CASE_KINDS
    }
    if (
        len(rows) != 12
        or set(family_counts.values()) != {4}
        or set(case_kind_counts.values()) != {3}
        or len({row["task_key"] for row in rows}) != 12
    ):
        raise ValueError("summary revision set is not balanced and unique")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "child",
        "objective": OBJECTIVE,
        "rows": len(rows),
        "family_counts": family_counts,
        "case_kind_counts": case_kind_counts,
        "case_records": case_records,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": [
            {"path": str(source_path), "sha256": sha256_file(source_path)}
        ],
        "source_trace_id": source_trace_id,
        "answer_free": False,
        "authored_reference": True,
        "native_prime_agent_context": True,
        "prime_agent_tools_available": True,
        "direct_text_response_local_to_revision": True,
        "assistant_target_messages_per_row": 1,
        "context_assistant_messages_per_row": 0,
        "failed_reasoning_tokens_in_targets": False,
        "word_counter": "markdown_bullet_regex_then_python_str_split_v1",
        "development_document_id": DEVELOPMENT_DOCUMENT_ID,
        "development_chapter_ids": ["scope", "operations", "exceptions"],
        "fresh_confirmation_documents_reserved": True,
        "tool_call_format": "openai_function_v1",
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                traces=[path.resolve() for path in args.traces],
                output_dir=args.output_dir.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
