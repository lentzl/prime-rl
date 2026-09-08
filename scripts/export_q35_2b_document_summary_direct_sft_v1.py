#!/usr/bin/env python3
"""Build direct native-summary episodes from reviewed chapters and retained TRAIN cases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from datasets import Dataset
from document_summary_evidence_training_v1 import training_chapters
from export_q35_2b_document_decision_sft_v1 import _wire_message, sha256_file
from export_q35_2b_document_summary_evidence_sft_v1 import ROOT, _action, _done, _result
from export_q35_2b_document_summary_live_revision_sft_v2 import _load_fixture_module

SCHEMA_VERSION = "qwen35-2b-document-summary-direct-sft/v1"
OBJECTIVE = "grounded_english_direct_chapter_key_bullets"
REPO = Path(__file__).resolve().parents[1]


def _context(trace_path: Path):
    traces = [
        t for line in trace_path.read_text().splitlines() if line.strip() for t in json.loads(line).get("traces", [])
    ]
    if len(traces) != 1:
        raise ValueError("need one observed native direct-summary trace")
    trace = traces[0]
    data = trace["task"]["data"]
    context = [_wire_message(copy.deepcopy(n["message"])) for n in trace["nodes"][:2]]
    if (
        trace["task"]["type"] != "DocumentSummaryEvidenceTask"
        or trace.get("errors")
        or data.get("direct_summary") is not True
        or trace.get("info", {}).get("summary_workflow") != "direct"
        or [m["role"] for m in context] != ["user", "user"]
        or [t["name"] for t in trace["tools"]] != ["ipython"]
        or not context[1]["content"].startswith(data["prompt"])
        or data["system_prompt"] not in context[0]["content"]
    ):
        raise ValueError("trace does not expose the direct native runtime and task context")
    budget = int(sum(len(p["text"].split()) for p in data["chapter"]["paragraphs"]) * 0.8)
    return trace, context, budget


def _chapters(source_dir: Path, teacher_path: Path, teacher_additions: Path | None = None):
    manifest = json.loads((source_dir / "SOURCES.json").read_text())
    labels = json.loads(teacher_path.read_text())
    label_sets = [labels]
    if teacher_additions is not None:
        label_sets.append(json.loads(teacher_additions.read_text()))
    if any(s.get("status") != f"complete_{len(s['chapters'])}_of_{len(s['chapters'])}_source_reviewed"
           for s in label_sets):
        raise ValueError("incomplete reviewed teacher labels")
    labels = {"chapters": [c for s in label_sets for c in s["chapters"]]}
    source_rows = {c["slug"]: c for c in manifest["chapters"]}
    if (
        manifest.get("split") != "TRAIN"
        or len(source_rows) < 20
        or len(source_rows) != len(manifest["chapters"])
        or len(labels["chapters"]) != len(source_rows)
        or {c["slug"] for c in labels["chapters"]} != set(source_rows)
        or {b["ebook"] for b in manifest["books"]} != {35, 120, 97, 37423}
    ):
        raise ValueError("incomplete reviewed public TRAIN corpus")
    public = []
    for label in labels["chapters"]:
        row = source_rows[label["slug"]]
        raw = (source_dir / f"{label['slug']}.md").read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["source_sha256"] or not label["review_points"]:
            raise ValueError(f"unreviewed or changed chapter: {label['slug']}")
        public.append(
            {
                "slug": label["slug"],
                "family": "public_chapter",
                "source_sha256": row["source_sha256"],
                "paragraphs": [{"text": p} for p in raw.decode().strip().split("\n\n")],
                "summary": "\n".join(f"- {b}" for b in label["bullets"]) + "\n",
            }
        )
    retained = [dict(c, family="retained_train") for c in training_chapters()]
    if len(retained) != 20:
        raise ValueError("expected twenty retained TRAIN cases")
    return [c for pair in zip_longest(retained, public) for c in pair if c is not None]


def _format_repair_feedback(trace):
    feedback = {
        _wire_message(node["message"])["content"]
        for node in trace["nodes"]
        if node["message"]["role"] == "user"
        and _wire_message(node["message"])["content"].startswith(
            "Chapter summarization: next file step.\nRewrite summary.md as only 3-5 Markdown bullet lines,"
        )
    }
    if len(feedback) != 1:
        raise ValueError("need one distinct observed model-visible bullet-rewrite instruction")
    return feedback.pop()


def _messages(context, original_budget, chapter, repair_feedback=None, *,
              repair_draft=None, repair_reasoning=None):
    paragraphs = chapter["paragraphs"]
    source = "\n\n".join(f"[chapter-p{i:03d}] {p['text']}" for i, p in enumerate(paragraphs, 1)) + "\n"
    summary = chapter["summary"]
    budget = int(sum(len(p["text"].split()) for p in paragraphs) * 0.8)
    messages = copy.deepcopy(context)
    old = f"at most {original_budget} total words"
    if messages[1]["content"].count(old) != 1:
        raise ValueError("observed prompt has no unique word allowance")
    messages[1]["content"] = messages[1]["content"].replace(old, f"at most {budget} total words")
    messages += [
        _action(
            "read-source",
            f"from pathlib import Path\nsource_text = Path('{ROOT}/source.md').read_text(encoding='utf-8')\nsource_text",
            "Read the complete chapter before choosing its main points.",
        ),
        _result("read-source", repr(source)),
    ]
    if repair_feedback is not None:
        draft = (repair_draft if repair_draft is not None else
                 " ".join(" ".join(p["text"] for p in paragraphs).split()[:350]) + "\n")
        draft_call_id = "write-summary-draft" if repair_draft is not None else "write-prose-draft"
        draft_action = _action(
            draft_call_id,
            f"draft_text = {draft!r}\nPath('{ROOT}/summary.md').write_text(draft_text, encoding='utf-8')",
            "Write a draft before returning." if repair_draft is not None else "Write a prose draft before returning.",
        )
        draft_done = _done()
        draft_action["trainable"] = draft_done["trainable"] = False
        messages += [
            draft_action,
            _result(draft_call_id, str(len(draft))),
            draft_done,
            {"role": "user", "content": repair_feedback},
        ]
    messages += [
        _action(
            "write-summary",
            f"summary_text = {summary!r}\nPath('{ROOT}/summary.md').write_text(summary_text, encoding='utf-8')",
            repair_reasoning or (
                "Replace the prose draft with the chapter's key points, not paragraph-by-paragraph copying. "
                if repair_feedback is not None else ""
            ) + "Select the main ideas or events, retaining essential qualifications and chronology. Keep possibilities distinct from actual events, use only this chapter, and write concise English bullets.",
        ),
        _result("write-summary", str(len(summary))),
        _done(),
    ]
    for message in messages:
        if message["role"] == "assistant":
            message.setdefault("trainable", True)
    return messages, source


def _semantic_repairs(path, chapters, cases):
    if path is None:
        return []
    specification = json.loads(path.read_text())
    if (specification.get("schema_version") != "document-summary-semantic-repairs/v1"
            or specification.get("split") != "TRAIN"
            or specification.get("status") != "source_reviewed"
            or not specification.get("feedback", "").strip()):
        raise ValueError("semantic repairs need a reviewed TRAIN specification and explicit feedback")
    by_slug = {chapter["slug"]: (chapter, case) for chapter, case in zip(chapters, cases, strict=True)}
    repairs, seen = [], set()
    for item in specification["cases"]:
        slug = item["base_slug"]
        if slug in seen or slug not in by_slug:
            raise ValueError("semantic repair needs one distinct existing TRAIN chapter")
        seen.add(slug)
        chapter, case = by_slug[slug]
        if item["source_sha256"] != case["source_sha256"] or not item["correction_reasoning"].strip():
            raise ValueError(f"semantic repair source/review mismatch: {slug}")
        draft = chapter["summary"]
        for change in item["changes"]:
            correct, incorrect = change["correct"], change["incorrect"]
            if not correct or not incorrect or correct == incorrect or draft.count(correct) != 1:
                raise ValueError(f"semantic draft edit must match exactly once: {slug}")
            draft = draft.replace(correct, incorrect, 1)
        lines = draft.splitlines()
        counts = [len(line[2:].split()) for line in lines]
        if (draft == chapter["summary"] or not 3 <= len(lines) <= 5
                or not all(line.startswith("- ") for line in lines)
                or not all(5 <= count <= 45 for count in counts)
                or sum(counts) > int(sum(len(p["text"].split()) for p in chapter["paragraphs"]) * 0.8)):
            raise ValueError(f"semantic draft must be changed but format-valid: {slug}")
        repairs.append((chapter, case, draft, specification["feedback"], item["correction_reasoning"]))
    if not repairs:
        raise ValueError("semantic repair specification contains no cases")
    return repairs


def export(*, trace_path: Path, source_dir: Path, teacher_path: Path, output_dir: Path,
           teacher_additions: Path | None = None, include_format_repairs: bool = False,
           semantic_repairs: Path | None = None):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    trace, context, original_budget = _context(trace_path)
    repair_feedback = _format_repair_feedback(trace) if include_format_repairs else None
    fixture = _load_fixture_module()
    excluded = {
        p["text"].strip()
        for doc, _ in (fixture.build_fixture(), fixture.build_confirmation_fixture())
        for chapter in doc["chapters"]
        for p in chapter["paragraphs"]
    }
    probe_dir = REPO / "experiments/qwen35-2b-document-summary-prime-agent-v1/chapter-probes"
    for path in [probe_dir / "city-shade.md", probe_dir / "evening-access.md", *probe_dir.glob("*-ch[0-9]*.md")]:
        excluded.update(path.read_text().strip().split("\n\n"))
    rows, cases = [], []
    chapters = _chapters(source_dir, teacher_path, teacher_additions)
    for chapter in chapters:
        paragraphs, summary = chapter["paragraphs"], chapter["summary"]
        lines = summary.splitlines()
        counts = [len(line[2:].split()) for line in lines]
        if (
            not 3 <= len(lines) <= 5
            or not all(line.startswith("- ") for line in lines)
            or not all(5 <= n <= 45 for n in counts)
            or sum(counts) > int(sum(len(p["text"].split()) for p in paragraphs) * 0.8)
            or any(p["text"].strip() in excluded for p in paragraphs)
        ):
            raise ValueError(f"invalid or evaluation-overlapping training case: {chapter['slug']}")
        messages, source = _messages(context, original_budget, chapter)
        if len(repr(source).encode()) >= 50 * 1024:
            raise ValueError("source observation exceeds conservative native tool-output allowance")
        rows.append(
            {
                "messages": messages,
                "tools": json.dumps(trace["tools"], sort_keys=True),
                "task_key": f"summary-direct-{chapter['slug']}",
                "trace_id": f"summary-direct-authored:{chapter['slug']}",
                "family": chapter["family"],
                "role": "child",
                "objective": OBJECTIVE,
            }
        )
        cases.append(
            {
                "slug": chapter["slug"],
                "family": chapter["family"],
                "source": source,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "summary": summary,
            }
        )
    reviewed_repairs = _semantic_repairs(semantic_repairs, chapters, cases)
    if include_format_repairs:
        for chapter, base_case in zip(chapters, list(cases), strict=True):
            slug = f"{chapter['slug']}-format-repair"
            messages, source = _messages(context, original_budget, chapter, repair_feedback)
            rows.append({
                "messages": messages,
                "tools": json.dumps(trace["tools"], sort_keys=True),
                "task_key": f"summary-direct-{slug}",
                "trace_id": f"summary-direct-authored:{slug}",
                "family": "format_repair",
                "role": "child",
                "objective": OBJECTIVE,
            })
            cases.append(dict(base_case, slug=slug, base_slug=chapter["slug"], family="format_repair"))
    for chapter, base_case, draft, feedback, reasoning in reviewed_repairs:
        slug = f"{chapter['slug']}-semantic-repair"
        messages, source = _messages(context, original_budget, chapter, feedback,
                                     repair_draft=draft, repair_reasoning=reasoning)
        rows.append({
            "messages": messages,
            "tools": json.dumps(trace["tools"], sort_keys=True),
            "task_key": f"summary-direct-{slug}",
            "trace_id": f"summary-direct-authored:{slug}",
            "family": "semantic_repair", "role": "child", "objective": OBJECTIVE,
        })
        cases.append(dict(base_case, slug=slug, base_slug=chapter["slug"], family="semantic_repair",
                          incorrect_draft=draft, correction_reasoning=reasoning,
                          feedback_kind="authored_user_revision_request_not_native_gate_feedback"))
    if len({r["task_key"] for r in rows}) != len(rows):
        raise ValueError("expected distinct direct-summary episodes")
    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    (output_dir / "CASES.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "child",
        "objective": OBJECTIVE,
        "rows": len(rows),
        "family_counts": dict(Counter(r["family"] for r in rows)),
        "answer_free": False,
        "tool_call_format": "openai_function_v1",
        "renderer_enable_thinking": True,
        "native_prime_agent_context": True,
        "direct_summary": True,
        "notes_stage": False,
        "trajectory_kind": "authored_teacher_episode_not_on_policy_replay",
        "independent_human_gold": False,
        "teacher_tool_results": "regenerated_from_authored_file_contents",
        "assistant_only_loss": True,
        "format_repair_episodes": len(chapters) if include_format_repairs else 0,
        "incorrect_draft_and_stop_masked": include_format_repairs or bool(reviewed_repairs),
        "semantic_repair_episodes": len(reviewed_repairs),
        "semantic_repairs_sha256": None if semantic_repairs is None else sha256_file(semantic_repairs),
        "semantic_repair_feedback_kind": (
            "authored_user_revision_request_not_native_gate_feedback" if reviewed_repairs else None
        ),
        "format_repair_feedback_sha256": (
            hashlib.sha256(repair_feedback.encode()).hexdigest() if repair_feedback is not None else None
        ),
        "format_repair_draft": "first_350_source_words_as_unmarked_prose" if include_format_repairs else None,
        "context_trace": {"path": str(trace_path), "sha256": sha256_file(trace_path), "trace_id": trace["id"]},
        "source_manifest_sha256": sha256_file(source_dir / "SOURCES.json"),
        "teacher_labels_sha256": sha256_file(teacher_path),
        "teacher_additions_sha256": None if teacher_additions is None else sha256_file(teacher_additions),
        "cases_sha256": sha256_file(output_dir / "CASES.json"),
        "eval_books_excluded_as_sources": [11, 2274, 37134],
        "incidental_overlap": "Dewey chapter 8 alludes to Alice's cake; no zero-phrase-overlap claim",
        "pretraining_contamination_possible": True,
        "broad_skill_claim": False,
        "dataset": {"path": "train.parquet", "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--teacher-labels", type=Path, required=True)
    parser.add_argument("--teacher-additions", type=Path)
    parser.add_argument("--include-format-repairs", action="store_true")
    parser.add_argument("--semantic-repairs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                trace_path=args.trace,
                source_dir=args.source_dir,
                teacher_path=args.teacher_labels,
                teacher_additions=args.teacher_additions,
                include_format_repairs=args.include_format_repairs,
                semantic_repairs=args.semantic_repairs,
                output_dir=args.output_dir,
            ),
            indent=2,
        )
    )
