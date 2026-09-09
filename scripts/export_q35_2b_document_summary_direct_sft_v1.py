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
NATIVE_REVISION_FAMILIES = {"native_format_repair", "native_semantic_repair", "native_write_type_repair"}
NATIVE_FAMILIES = {"native_child", "native_count_repair", "native_interruption_repair"} | NATIVE_REVISION_FAMILIES


def _native_context(path):
    from document_summary_v1.taskset import DocumentSummaryConfig, DocumentSummaryTaskset

    traces = [t for line in path.read_text().splitlines() if line.strip()
              for t in json.loads(line).get("traces", [])]
    if len(traces) != 1 or traces[0].get("errors"):
        raise ValueError("need one captured native owner/child trace for interface context")
    trace = traces[0]
    if (trace["task"]["type"] != "DocumentSummaryMarkdownTask"
            or [tool["name"] for tool in trace["tools"]] != ["ipython"]):
        raise ValueError("not the native Markdown owner/child interface")
    current = DocumentSummaryTaskset(DocumentSummaryConfig(mode="owner_direct")).load()[0]
    for call in trace["calls"]:
        index = call["node"]
        if index < 2:
            continue
        context = [_wire_message(copy.deepcopy(n["message"])) for n in trace["nodes"][index - 2:index]]
        if ([m["role"] for m in context] != ["user", "user"]
                or "\nRecursive agent depth: 1\n" not in context[0]["content"]
                or not context[1]["content"].startswith("[task from parent]\n\n")
                or call["client_session_id"] not in context[0]["content"]):
            continue
        old = trace["task"]["data"]["system_prompt"]
        if context[0]["content"].count(old) != 1:
            raise ValueError("child does not expose the inherited task instruction")
        context[0]["content"] = context[0]["content"].replace(old, current.data.system_prompt, 1)
        return context[0], trace["tools"], {
            "path": str(path), "sha256": sha256_file(path), "trace_id": trace["id"],
            "client_session_id": call["client_session_id"],
            "inherited_task_instruction_replaced_with_current_role_aware_prompt": True,
            "evaluation_task_and_source_excluded": True,
        }
    raise ValueError("no observed depth-one child runtime prefix")


def _native_messages(runtime, chapter, base_case, index, *, count_repair=False,
                     revision_kind=None, repair_draft=None, repair_reasoning=None):
    from document_summary_v1.taskset import MARKDOWN_CHILD_RECOVERY_FEEDBACK, _markdown_jobs
    from export_q35_2b_document_summary_owner_sft_v1 import _reply, _tool

    chapter_id = f"chapter-{1 + index % 4:03d}"
    job = next(iter(_markdown_jobs({"chapters": [{
        "id": chapter_id, "title": chapter["slug"], "paragraphs": chapter["paragraphs"],
    }]}).values()))
    summary, source = chapter["summary"], base_case["source"]
    budget = int(sum(len(p["text"].split()) for p in chapter["paragraphs"]) * 0.8)
    words = sum(len(line[2:].split()) for line in summary.splitlines())
    if revision_kind is not None and (
            revision_kind not in NATIVE_REVISION_FAMILIES or count_repair
            or not repair_draft or repair_draft == summary or not repair_reasoning):
        raise ValueError("native revision needs a distinct draft and reviewed correction, without a count repair")
    if count_repair and not words <= budget < len(summary):
        raise ValueError("count correction needs a valid word count exceeding the allowance only in characters")
    write_code = (f"summary_text = {summary!r}\ncharacters_written = "
                  f"Path({job['summary_path']!r}).write_text(summary_text, encoding='utf-8')\n"
                  "print(characters_written)")
    messages = [copy.deepcopy(runtime), {"role": "user", "content": "[task from parent]\n\n" + job["prompt"]},
        _tool("read-source", "from pathlib import Path\n"
              f"source_text = Path({job['source_path']!r}).read_text(encoding='utf-8')\n"
              "print(source_text, end='')", "Read the complete assigned source before selecting the key points."),
        _result("read-source", source),
    ]
    if revision_kind is not None:
        write_type_repair = revision_kind == "native_write_type_repair"
        draft_code = (f"draft_text = {repair_draft!r}\n"
                      f"print(Path({job['summary_path']!r}).write_text(draft_text, encoding='utf-8'))")
        draft_observation = str(len(repair_draft)) + "\n"
        inspect_code = ""
        if write_type_repair:
            draft_code = (f"notes_text = Path({job['summary_path']!r}).write_text({repair_draft!r}, encoding='utf-8')\n"
                          f"Path({job['summary_path']!r}).write_text(notes_text, encoding='utf-8')")
            draft_observation = "TypeError: data must be str, not int\n"
            inspect_code = "print(type(notes_text).__name__)\n"
        messages += [
            _tool("write-summary-draft", draft_code,
                  "Save this draft as the chapter summary.", trainable=False),
            _result("write-summary-draft", draft_observation),
            _tool("read-saved-draft", inspect_code + f"saved_draft = Path({job['summary_path']!r}).read_text(encoding='utf-8')\n"
                  "print(saved_draft, end='')",
                  ("The second write rejected an integer. Earlier statements in the cell may have succeeded. "
                   "Inspect notes_text's actual type and read the assigned saved file; do not repeat the failing write. "
                   if write_type_repair else "") +
                  "Before reporting completion, inspect what was saved against the complete source already read. "
                  "A successful write does not establish useful key bullets or factual coverage."),
            _result("read-saved-draft", ("int\n" if write_type_repair else "") + repair_draft),
        ]
    messages += [
        _tool("write-summary", write_code,
              repair_reasoning or "Select the main ideas from this chapter, preserve qualifications and event order, "
              "and write 3-5 concise English bullets. Do not supply events or conclusions from outside the source."),
        _result("write-summary", str(len(summary)) + "\n"),
    ]
    if count_repair:
        messages += [
            _tool("repeat-write", write_code,
                  "The returned number is above the word allowance. Rewrite the summary again.", trainable=False),
            _result("repeat-write", str(len(summary)) + "\n\n" + MARKDOWN_CHILD_RECOVERY_FEEDBACK),
            _tool("check-words", "word_count = sum(len(line[2:].split()) for line in summary_text.splitlines())\n"
                  "print(word_count)", "The previous comparison confused characters with words. "
                  "Path.write_text succeeded and returned characters written. Count the actual bullet words, "
                  "excluding Markdown markers; do not rewrite unchanged text because of its character count."),
            _result("check-words", str(words) + "\n"),
        ]
    receipt = {"chapter_id": chapter_id, "summary_path": job["summary_path"]}
    messages += [
        _tool("send-receipt", f"import json\nreceipt = {receipt!r}\n"
              "delivery = await agent_message.send(json.dumps(receipt), receiver_role='parent')\n"
              "print(delivery['deliveryStatus'])",
              "The assigned source-grounded bullets are saved. The write return counts characters, not words. "
              "Send this chapter's exact saved-file receipt once; do not send the summary text or another path."),
        _result("send-receipt", "queued\n"),
        _reply("Done.", "The native send succeeded and queued the receipt. That does not prove the owner "
               "has consumed it. My assigned work is finished; stop without resending, polling or rewriting."),
    ]
    family = revision_kind or ("native_count_repair" if count_repair else "native_child")
    case = dict(base_case, slug=f"{chapter['slug']}-{family.replace('_', '-')}",
                base_slug=chapter["slug"], family=family, native_job=job,
                word_budget=budget, summary_word_count=words,
                receipt_observation="scripted_queued_status_not_live_delivery",
                masked_message_indices=[4] if revision_kind else ([6] if count_repair else []))
    if revision_kind is not None:
        case.update(incorrect_draft=repair_draft, correction_reasoning=repair_reasoning,
                    revision_provenance="authored_self_review_before_receipt_not_native_gate_feedback")
        if revision_kind == "native_write_type_repair":
            case["exception_observation"] = "replayed_exception_type_and_message_not_full_native_traceback"
    return messages, case


def _native_interruption_messages(runtime, chapter, base_case, index):
    from document_summary_v1.taskset import MARKDOWN_CHILD_RECOVERY_FEEDBACK
    from export_q35_2b_document_summary_owner_sft_v1 import _tool

    baseline, case = _native_messages(runtime, chapter, base_case, index)
    job = case["native_job"]
    parent_message = "[from parent]\n\n" + "\n".join(str(41 + index + offset) for offset in range(3))
    recovery = "This IPython call contained no executable code. " + MARKDOWN_CHILD_RECOVERY_FEEDBACK
    reasoning = (
        "The numeric parent message adds no change to my assigned source or output requirements. "
        "Line counts do not show the chapter's content, and the empty cell wrote nothing. "
        "The recovery's conditional stop is not evidence that a write or send happened. "
        "Inspect the assigned output's existence and display the complete source before selecting key points."
    )
    messages = baseline[:2] + [
        _tool("load-source-counts", "from pathlib import Path\n"
              f"source_text = Path({job['source_path']!r}).read_text(encoding='utf-8')\n"
              "print(len(source_text.splitlines()))",
              "The line count is enough to read this chapter.", trainable=False),
        _result("load-source-counts", str(len(case["source"].splitlines())) + "\n"),
        {"role": "user", "content": parent_message},
        _tool("empty-progress", "", "Wait for the parent's progress numbers to stop.", trainable=False),
        _result("empty-progress", recovery),
        _tool("read-source", f"summary_file = Path({job['summary_path']!r})\n"
              "print(summary_file.exists())\n"
              f"source_text = Path({job['source_path']!r}).read_text(encoding='utf-8')\n"
              "print(source_text, end='')", reasoning),
        _result("read-source", "False\n" + case["source"]),
        *baseline[4:6],
        _tool("verify-saved-summary", "saved_summary = summary_file.read_text(encoding='utf-8')\n"
              "print(saved_summary, end='')",
              "The write returned a character count. Read the saved file to confirm the actual summary "
              "before sending its receipt; neither a progress message nor a conditional instruction proves completion."),
        _result("verify-saved-summary", case["summary"]),
        *baseline[6:],
    ]
    case.update(slug=f"{chapter['slug']}-native-interruption-repair", family="native_interruption_repair",
                masked_message_indices=[2, 5], parent_message=parent_message,
                correction_reasoning=reasoning, empty_call_feedback=recovery,
                interruption_provenance="authored_TRAIN_numeric_message_and_empty_call_recovery",
                parent_message_observation="simplified_parent_message_without_native_envelope_ids")
    return messages, case


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
    book_ids = {b["ebook"] for b in manifest["books"]}
    if (
        manifest.get("split") != "TRAIN"
        or len(source_rows) < 20
        or len(source_rows) != len(manifest["chapters"])
        or len(labels["chapters"]) != len(source_rows)
        or {c["slug"] for c in labels["chapters"]} != set(source_rows)
        or not {35, 120, 97, 37423} <= book_ids <= {35, 120, 97, 37423, 769}
        or len(book_ids) != len(manifest["books"])
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
           semantic_repairs: Path | None = None, native_child_trace: Path | None = None,
           include_native_revisions: bool = False, include_native_write_repairs: bool = False,
           include_native_interruption_repairs: bool = False):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if (include_native_revisions or include_native_write_repairs or include_native_interruption_repairs) and native_child_trace is None:
        raise ValueError("native revisions require the observed child interface")
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
    base_cases = list(cases)
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
    native_context = None
    if native_child_trace is not None:
        runtime, native_tools, native_context = _native_context(native_child_trace)
        if native_tools != trace["tools"]:
            raise ValueError("direct and native-child IPython schemas differ")
        for index, (chapter, base_case) in enumerate(zip(chapters, base_cases, strict=True)):
            budget = int(sum(len(p["text"].split()) for p in chapter["paragraphs"]) * 0.8)
            for repair in ([False, True] if len(chapter["summary"]) > budget else [False]):
                messages, case = _native_messages(runtime, chapter, base_case, index, count_repair=repair)
                rows.append({
                    "messages": messages, "tools": json.dumps(native_tools, sort_keys=True),
                    "task_key": f"summary-direct-{case['slug']}",
                    "trace_id": f"summary-direct-authored:{case['slug']}",
                    "family": case["family"], "role": "child", "objective": OBJECTIVE,
                })
                cases.append(case)
        if include_native_revisions or include_native_write_repairs:
            native_revisions = [
                (chapter, base_case,
                 " ".join(" ".join(p["text"] for p in chapter["paragraphs"]).split()[:350]) + "\n",
                 "The saved prose copies source text instead of selecting the chapter's key points. "
                 "Use the complete source, not just its opening; preserve main conclusions or event order and "
                 "essential qualifications. Replace it with 3-5 concise, source-grounded English bullets.",
                 "native_format_repair")
                for chapter, base_case in zip(chapters, base_cases, strict=True) if include_native_revisions
            ]
            native_revisions += [(chapter, base_case, draft, reasoning, "native_semantic_repair")
                                 for chapter, base_case, draft, _, reasoning in reviewed_repairs if include_native_revisions]
            if include_native_write_repairs:
                native_revisions += [
                    (chapter, base_case,
                     " ".join(" ".join(p["text"] for p in chapter["paragraphs"]).split()[:350]) + "\n",
                     "The file contains the first write's prose draft even though the second write failed. "
                     "notes_text holds the integer character count returned by write_text, not summary text. "
                     "Use a separate string for the corrected 3-5 English bullets and retain the count separately. "
                     "Select the key points from the complete source, preserving qualifications and chronology; "
                     "a successful draft save is not evidence of source-grounded summarization.",
                     "native_write_type_repair")
                    for chapter, base_case in zip(chapters, base_cases, strict=True)
                ]
            indices = {chapter["slug"]: index for index, chapter in enumerate(chapters)}
            for chapter, base_case, draft, reasoning, family in native_revisions:
                messages, case = _native_messages(
                    runtime, chapter, base_case, indices[chapter["slug"]],
                    revision_kind=family, repair_draft=draft, repair_reasoning=reasoning)
                rows.append({
                    "messages": messages, "tools": json.dumps(native_tools, sort_keys=True),
                    "task_key": f"summary-direct-{case['slug']}",
                    "trace_id": f"summary-direct-authored:{case['slug']}",
                    "family": family, "role": "child", "objective": OBJECTIVE,
                })
                cases.append(case)
        if include_native_interruption_repairs:
            for index, (chapter, base_case) in enumerate(zip(chapters, base_cases, strict=True)):
                messages, case = _native_interruption_messages(runtime, chapter, base_case, index)
                rows.append({
                    "messages": messages, "tools": json.dumps(native_tools, sort_keys=True),
                    "task_key": f"summary-direct-{case['slug']}",
                    "trace_id": f"summary-direct-authored:{case['slug']}",
                    "family": case["family"], "role": "child", "objective": OBJECTIVE,
                })
                cases.append(case)
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
    if native_context is not None:
        manifest.update(
            native_child_context=native_context,
            native_child_episodes=sum(c["family"] == "native_child" for c in cases),
            native_count_repair_episodes=sum(c["family"] == "native_count_repair" for c in cases),
            native_receipt_observations="scripted_queued_status_not_live_delivery",
            native_count_repair_feedback="current_task_interception_feedback_after_repeated_success",
            incorrect_native_retry_masked=True,
        )
        if include_native_revisions or include_native_write_repairs:
            manifest.update(
                native_format_repair_episodes=len(chapters) if include_native_revisions else 0,
                native_semantic_repair_episodes=len(reviewed_repairs) if include_native_revisions else 0,
                native_revision_provenance="authored_self_review_before_receipt_not_native_gate_feedback",
                incorrect_native_draft_masked=True,
            )
        if include_native_write_repairs:
            manifest.update(
                native_write_type_repair_episodes=len(chapters),
                native_write_type_repair_provenance="authored_TRAIN_analogue_of_partial_execution_and_integer_shadowing",
                native_write_type_repair_observation="replayed_exception_type_and_message_not_full_native_traceback",
                incorrect_native_draft_masked=True,
            )
        if include_native_interruption_repairs:
            manifest.update(
                native_interruption_repair_episodes=len(chapters),
                native_interruption_provenance="authored_TRAIN_numeric_message_and_empty_call_recovery",
                native_interruption_parent_message="simplified_parent_message_without_native_envelope_ids",
                native_interruption_feedback="current_task_empty_ipython_recovery_not_a_completion_assertion",
                incorrect_native_interruption_masked=True,
            )
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
    parser.add_argument("--native-child-trace", type=Path)
    parser.add_argument("--include-native-revisions", action="store_true")
    parser.add_argument("--include-native-write-repairs", action="store_true")
    parser.add_argument("--include-native-interruption-repairs", action="store_true")
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
                native_child_trace=args.native_child_trace,
                include_native_revisions=args.include_native_revisions,
                include_native_write_repairs=args.include_native_write_repairs,
                include_native_interruption_repairs=args.include_native_interruption_repairs,
                output_dir=args.output_dir,
            ),
            indent=2,
        )
    )
