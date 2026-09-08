#!/usr/bin/env python3
"""Export authored source/notes/realization episodes in observed Prime Agent context."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from datasets import Dataset
from document_summary_evidence_training_v1 import training_chapters
from export_q35_2b_document_decision_sft_v1 import _wire_message, sha256_file
from export_q35_2b_document_summary_live_revision_sft_v2 import _load_fixture_module

SCHEMA_VERSION = "qwen35-2b-document-summary-evidence-sft/v1"
OBJECTIVE = "grounded_english_source_notes_summary_episode"
ROOT = "/workspace/document-summary-v1"


def _text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content)


def _context(trace_path):
    traces = [
        trace
        for line in trace_path.read_text().splitlines()
        if line.strip()
        for trace in json.loads(line).get("traces", [])
        if trace.get("task", {}).get("type") == "DocumentSummaryEvidenceTask"
    ]
    if len(traces) != 1 or traces[0].get("errors"):
        raise ValueError("need exactly one error-free evidence episode")
    trace = traces[0]
    messages = [node["message"] for node in trace["nodes"]]
    feedback = [m for m in messages if m["role"] == "user" and "Your notes are now saved" in _text(m)]
    model_visible = [m for m in feedback if _text(m).startswith("Chapter summarization: next file step.")]
    if model_visible:
        feedback = model_visible
    elif trace.get("info", {}).get("evidence_feedback_rewrites"):
        raise ValueError("rewritten model-visible continuation is absent from the trace")
    if len(feedback) != 1 or [tool["name"] for tool in trace["tools"]] != ["ipython"]:
        raise ValueError("missing native notes continuation or IPython schema")
    return trace, messages[:2], feedback[0]


def _action(call_id, code, reasoning):
    compile(code, "teacher-ipython", "exec")
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "ipython",
                    "arguments": json.dumps({"code": code}, separators=(",", ":")),
                },
            }
        ],
    }


def _result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _done():
    return {
        "role": "assistant",
        "content": "Done.",
        "reasoning_content": "The file write succeeded; stop now.",
        "tool_calls": [],
    }


def _messages(context, feedback, chapter, original_budget):
    source = "\n\n".join(f"[{p['id']}] {p['text']}" for p in chapter["paragraphs"]) + "\n"
    notes, summary = chapter["notes"], chapter["summary"]
    word_budget = int(sum(len(p["text"].split()) for p in chapter["paragraphs"]) * 0.8)
    next_phase = _text(feedback).replace(f"at most {original_budget} total words", f"at most {word_budget} total words")
    if f"at most {word_budget} total words" not in next_phase:
        raise ValueError("observed continuation does not contain the expected budget")
    read_source = (
        f"from pathlib import Path\nsource_text = Path('{ROOT}/source.md').read_text(encoding='utf-8')\nsource_text"
    )
    write_notes = f"notes_text = {notes!r}\nPath('{ROOT}/notes.md').write_text(notes_text, encoding='utf-8')"
    read_notes = f"notes_text = Path('{ROOT}/notes-extracted.md').read_text(encoding='utf-8')\nnotes_text"
    write_summary = f"summary_text = {summary!r}\nPath('{ROOT}/summary.md').write_text(summary_text, encoding='utf-8')"
    return [
        *[_wire_message(copy.deepcopy(m)) for m in context],
        _action("read-source", read_source, "Read the source before selecting its obligations."),
        _result("read-source", repr(source)),
        _action(
            "write-notes",
            write_notes,
            "Preserve actors, conditions and qualifications in source-linked notes, without imposing the final word limit.",
        ),
        _result("write-notes", str(len(notes))),
        _done(),
        {"role": "user", "content": next_phase},
        _action("read-notes", read_notes, "Read the saved notes before realizing the summary."),
        _result("read-notes", repr(notes)),
        _action(
            "write-summary",
            write_summary,
            "Write concise Markdown bullets while retaining all recorded fields, linked actions and qualifiers.",
        ),
        _result("write-summary", str(len(summary))),
        _done(),
    ]


def export(*, trace_path: Path, output_dir: Path):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    trace, context, feedback = _context(trace_path)
    original_budget = int(sum(len(p["text"].split()) for p in trace["task"]["data"]["chapter"]["paragraphs"]) * 0.8)
    rows, cases = [], []
    fixture = _load_fixture_module()
    excluded_sources = {
        paragraph["text"]
        for document, _ in (fixture.build_fixture(), fixture.build_confirmation_fixture())
        for chapter in document["chapters"]
        for paragraph in chapter["paragraphs"]
    }
    for chapter in training_chapters():
        if any(p["text"] in excluded_sources for p in chapter["paragraphs"]):
            raise ValueError("training source overlaps the Northstar/Cedar probes")
        bullets = [line[2:] for line in chapter["summary"].splitlines()]
        budget = int(sum(len(p["text"].split()) for p in chapter["paragraphs"]) * 0.8)
        if (
            len(bullets) != 4
            or not all(5 <= len(b.split()) <= 45 for b in bullets)
            or sum(len(b.split()) for b in bullets) > budget
            or any(b.casefold() == p["text"].casefold() for b in bullets for p in chapter["paragraphs"])
        ):
            raise ValueError(f"invalid summary length/format: {chapter['slug']}")
        episode = _messages(context, feedback, chapter, original_budget)
        for phase in ("extraction", "realization"):
            messages = copy.deepcopy(episode[:7] if phase == "extraction" else episode)
            for index, message in enumerate(messages):
                if message["role"] == "assistant":
                    message["trainable"] = phase == "extraction" or index >= 8
            rows.append(
                {
                    "messages": messages,
                    "tools": json.dumps(trace["tools"], sort_keys=True, separators=(",", ":")),
                    "task_key": f"summary-evidence-{chapter['slug']}-{phase}",
                    "trace_id": f"summary-evidence-authored:{chapter['slug']}:{phase}",
                    "phase": phase,
                    "family": f"summary_{chapter['family']}",
                    "role": "child",
                    "objective": OBJECTIVE,
                }
            )
        cases.append(
            {
                "document_id": chapter["slug"],
                "family": chapter["family"],
                "source": chapter["paragraphs"],
                "teacher_notes": chapter["notes"],
                "teacher_summary": chapter["summary"],
                "word_budget": budget,
                "summary_words": sum(len(b.split()) for b in bullets),
            }
        )
    if len(rows) != 32 or len({r["task_key"] for r in rows}) != 32:
        raise ValueError("expected two TRAIN phases for each of sixteen documents")
    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "child",
        "objective": OBJECTIVE,
        "rows": len(rows),
        "family_counts": {
            family: sum(r["family"] == family for r in rows) for family in sorted({r["family"] for r in rows})
        },
        "answer_free": False,
        "authored_reference": True,
        "tool_call_format": "openai_function_v1",
        "native_prime_agent_context": True,
        "continuation_style": (
            "model_visible_file_step" if _text(feedback).startswith("Chapter summarization: next file step.")
            else "historical_generic_gate"
        ),
        "context_trace": {"path": str(trace_path), "sha256": sha256_file(trace_path), "trace_id": trace["id"]},
        "trajectory_kind": "authored_teacher_episode_not_on_policy_replay",
        "teacher_tool_results": "regenerated_from_authored_file_contents",
        "teacher_notes_training_only": True,
        "prior_phase_assistant_messages_trainable": False,
        "phase_counts": {"extraction": 16, "realization": 16},
        "renderer_enable_thinking": True,
        "distinct_documents": 16,
        "previous_training_sources_reused": 12,
        "new_contrast_documents": 4,
        "development_and_cedar_source_text_excluded": True,
        "broad_skill_claim": False,
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "CASES.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(trace_path=args.trace.resolve(), output_dir=args.output_dir.resolve()), indent=2))
