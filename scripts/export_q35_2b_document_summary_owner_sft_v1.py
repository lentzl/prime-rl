#!/usr/bin/env python3
"""Teach native document ownership with TRAIN-only scripted handoffs and rehearsal."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from datasets import Dataset
from document_summary_v1.taskset import (
    INDEX_PATH,
    MARKDOWN_OUTPUT_PATH,
    DocumentSummaryConfig,
    DocumentSummaryTaskset,
    _markdown_jobs,
)
from export_q35_2b_document_decision_sft_v1 import _wire_message, sha256_file

SCHEMA_VERSION = "qwen35-2b-document-summary-owner-sft/v1"
OBJECTIVE = "native_index_delegation_receipt_fanin_with_role_rehearsal"
REHEARSAL_SCHEMA = "qwen35-2b-adaptive-cognition-sft/v3"


def _tool(call_id, code, reasoning, *, trainable=True):
    compile(code, "owner-teacher", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    return {
        "role": "assistant", "content": "", "reasoning_content": reasoning,
        "trainable": trainable,
        "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": "ipython", "arguments": json.dumps({"code": code}),
        }}],
    }


def _result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _reply(content, reasoning):
    return {"role": "assistant", "content": content, "reasoning_content": reasoning,
            "trainable": True}


def _context(path):
    episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    traces = [trace for episode in episodes for trace in episode.get("traces", [])]
    if len(traces) != 1:
        raise ValueError("expected one native owner context trace")
    trace = traces[0]
    runtime = _wire_message(copy.deepcopy(trace["nodes"][0]["message"]))
    if (trace["task"]["type"] != "DocumentSummaryMarkdownTask" or trace.get("errors")
            or "Recursive agent depth: 0" not in runtime["content"]
            or trace["task"]["data"]["system_prompt"] not in runtime["content"]
            or [tool["name"] for tool in trace["tools"]] != ["ipython"]):
        raise ValueError("not the observed native owner/IPython context")
    task = DocumentSummaryTaskset(DocumentSummaryConfig(mode="owner_direct")).load()[0]
    runtime["content"] = runtime["content"].replace(
        trace["task"]["data"]["system_prompt"], task.data.system_prompt, 1
    )
    return [runtime, {"role": "user", "content": task.data.prompt}], trace["tools"]


def _case(chapters, document_number, repair, *, wait_repair=False):
    count = 1 + document_number % 4
    selected = [chapters[(document_number * 3 + i) % len(chapters)] for i in range(count)]
    if document_number % 2:
        selected.reverse()
    document = {"document_id": f"owner-train-{document_number:02d}", "chapters": [
        {"id": c["slug"], "title": c["slug"].replace("-", " ").title(),
         "paragraphs": [{"text": c["source"]}]} for c in selected
    ]}
    jobs = _markdown_jobs(document)
    index = {"document_id": document["document_id"], "output_path": MARKDOWN_OUTPUT_PATH,
             "chapters": list(jobs.values())}
    deliveries = []
    arrival_order = list(jobs)
    if document_number % 2:
        arrival_order.reverse()
    for worker in arrival_order:
        job = jobs[worker]
        payload = {"chapter_id": job["chapter_id"], "summary_path": job["summary_path"]}
        if repair and worker == arrival_order[0]:
            deliveries.append({"worker": worker, "payload": {
                **payload, "summary_path": "/workspace/document-summary-v1/unassigned.md",
            }, "valid": False})
        deliveries.append({"worker": worker, "payload": payload, "valid": True})
    return {
        "task_key": f"{document['document_id']}:{'wait-repair' if wait_repair else 'repair' if repair else 'clean'}",
        "family": ("owner_wait_repair" if wait_repair else
                   "owner_schema_receipt_repair" if repair else "owner_delegation_fanin"),
        "index": index, "schema_repair": repair, "deliveries": deliveries,
        "wait_repair": wait_repair,
        "wait_repair_after_receipts": (document_number % 2) if wait_repair else None,
        "chapters": selected,
        "summaries": {worker: c["summary"] for worker, c in zip(jobs, selected, strict=True)},
        "authorship": "scripted_training_episode_not_on_policy_or_observed_child_execution",
    }


def _poll_handles(names):
    return [
        _tool("poll-handles", "pending = sorted(handles)\nfor _ in range(2):\n"
              "    pending = sorted(handles)\nprint(json.dumps(pending))",
              "Wait for the children by repeatedly checking whether the handle dictionary empties.",
              trainable=False),
        _result("poll-handles", json.dumps(sorted(names)) + "\n"),
    ]


def _messages(context, case):
    messages = copy.deepcopy(context)
    messages += [
        _tool("read-index", f"import json\nfrom pathlib import Path\n"
              f"index = json.loads(Path({INDEX_PATH!r}).read_text(encoding='utf-8'))\nprint(json.dumps(index, indent=2))",
              "Read and display the index before using its fields. The source chapters belong to children."),
        _result("read-index", json.dumps(case["index"], indent=2) + "\n"),
    ]
    if case["schema_repair"]:
        messages += [
            _tool("wrong-schema", "try:\n    chapters = index['entries']\n"
                  "except KeyError as error:\n    print(f'KeyError: {error}')",
                  "Assume the jobs are in entries.", trainable=False),
            _result("wrong-schema", "KeyError: 'entries'\n"),
        ]
    names = [job["worker"] for job in case["index"]["chapters"]]
    messages += [
        _tool("spawn-chapters", "chapters = index['chapters']\nhandles = {}\nreceipts = {}\n"
              "for job in chapters:\n"
              "    handles[job['worker']] = await rlm(job['prompt'], name=job['worker'])\n"
              "print(json.dumps(list(handles)))",
              ("The failed lookup did not define chapters. The displayed dictionary has a chapters list, "
               "not entries; use that actual field instead of repeating the error. "
               if case["schema_repair"] else "") +
              "Keep each job's exact worker name and complete prompt together in index order. "
              "Retain every returned native handle; do not rewrite the assignments or sort separate lists."),
        _result("spawn-chapters", json.dumps(names) + "\n"),
    ]
    if case.get("wait_repair_after_receipts") == 0:
        messages += _poll_handles(names)
    messages += [
        _reply("Waiting for the named chapter workers' receipts.",
               ("Polling retained handles cannot await results: these handles remain in the dictionary "
                "after a child finishes. Do not turn this into a sleep loop or clear the handles. "
                if case.get("wait_repair_after_receipts") == 0 else "") +
               "All requested workers have been admitted. End this turn without polling or respawning. "
               "No summary can be read or assembled before the matching receipts arrive."),
    ]
    received = set()
    for i, delivery in enumerate(case["deliveries"]):
        worker, payload = delivery["worker"], delivery["payload"]
        raw = json.dumps(payload)
        messages.append({"role": "user", "content":
            f"[from child:{worker}]\nAgent-to-agent message received.\n"
            f"Source: agent_message\nFrom: {worker}\n\n{raw}"})
        if delivery["valid"]:
            received.add(worker)
        call_id = f"receipt-{i}"
        messages += [
            _tool(call_id, f"worker = {worker!r}\npayload = json.loads({raw!r})\n"
                  "job = next(job for job in chapters if job['worker'] == worker)\n"
                  "expected = {'chapter_id': job['chapter_id'], 'summary_path': job['summary_path']}\n"
                  "if payload == expected:\n    receipts[worker] = payload\nprint(json.dumps(sorted(receipts)))",
                  "Match both chapter identity and assigned path against the index. "
                  + ("This receipt matches; retain it in the persistent kernel."
                     if delivery["valid"] else
                     "This path does not match. Do not accept it or read that file; keep waiting for a valid receipt.")),
            _result(call_id, json.dumps(sorted(received)) + "\n"),
        ]
        if received != set(names):
            repair_wait = case.get("wait_repair_after_receipts") == len(received)
            if repair_wait:
                messages += _poll_handles(names)
            messages.append(_reply(
                "Waiting for matching receipts from " + ", ".join(n for n in names if n not in received) + ".",
                ("The handle dictionary includes finished as well as pending children; it does not empty "
                 "when messages arrive. Track received receipts separately and end the turn so new messages "
                 "can be handled. Do not wait in an IPython loop or discard a handle. " if repair_wait else "") +
                "The handoff is incomplete. Keep the existing handles and valid receipts; yield without polling, "
                "respawning, reading summaries or creating substitute content.",
            ))
    messages += [
        _tool("assemble", "expected_receipts = {job['worker']: {'chapter_id': job['chapter_id'], "
              "'summary_path': job['summary_path']} for job in chapters}\n"
              "assert receipts == expected_receipts\n"
              "document_text = '\\n\\n'.join(job['heading'] + '\\n\\n' + "
              "Path(job['summary_path']).read_text(encoding='utf-8').strip() "
              "for job in chapters) + '\\n'\n"
              "Path(index['output_path']).write_text(document_text, encoding='utf-8')\nprint(len(chapters))",
              "All named workers have supplied their matching receipts. Read only their assigned summaries. "
              "Assemble their wording unchanged under the index headings in index order, regardless of arrival order."),
        _result("assemble", str(len(names)) + "\n"),
        _reply(f"Saved {len(names)} chapter summaries to {MARKDOWN_OUTPUT_PATH}.",
               "The assembled document is saved. Return its path and chapter count, then stop calling tools."),
    ]
    return messages


def export(runtime_trace: Path, training_dir: Path, rehearsal_dir: Path, output_dir: Path,
           *, include_wait_repairs=False):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    context, tools = _context(runtime_trace)
    training_manifest = json.loads((training_dir / "MANIFEST.json").read_text())
    if (training_manifest["schema_version"] != "qwen35-2b-document-summary-direct-sft/v1"
            or sha256_file(training_dir / "CASES.json") != training_manifest["cases_sha256"]):
        raise ValueError("changed or unsupported TRAIN source cases")
    chapters = [c for c in json.loads((training_dir / "CASES.json").read_text())
                if c["family"] == "retained_train"]
    if len(chapters) != 20 or len({c["slug"] for c in chapters}) != 20:
        raise ValueError("expected the twenty existing reviewed TRAIN chapters")
    rehearsal_manifest = json.loads((rehearsal_dir / "MANIFEST.json").read_text())
    if (rehearsal_manifest["schema_version"] != REHEARSAL_SCHEMA
            or rehearsal_manifest["rows"] != 48
            or sha256_file(rehearsal_dir / "train.parquet") != rehearsal_manifest["dataset"]["sha256"]):
        raise ValueError("changed or unsupported acquired role rehearsal")
    rehearsal = list(Dataset.from_parquet(str(rehearsal_dir / "train.parquet")))
    cases = [_case(chapters, i, repair) for i in range(12) for repair in (False, True)]
    if include_wait_repairs:
        cases += [_case(chapters, i, False, wait_repair=True) for i in range(12)]
    owner_rows = [{"messages": _messages(context, c), "tools": json.dumps(tools),
                   "task_key": c["task_key"], "trace_id": c["task_key"], "family": c["family"],
                   "role": "coordinator", "objective": OBJECTIVE} for c in cases]
    rows = [row for triple in zip_longest(owner_rows, rehearsal[::2], rehearsal[1::2])
            for row in triple if row is not None]
    columns = dict.fromkeys(key for row in rows for key in row)
    rows = [{key: row.get(key) for key in columns} for row in rows]
    output_dir.mkdir(parents=True)
    Dataset.from_list(rows).to_parquet(str(output_dir / "train.parquet"))
    (output_dir / "CASES.json").write_text(json.dumps(cases, indent=2) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "complete", "role": "coordinator",
        "objective": OBJECTIVE, "rows": len(rows), "family_counts": dict(Counter(r["family"] for r in rows)),
        "task_keys": [r["task_key"] for r in rows], "answer_free": False,
        "tool_call_format": "openai_function_v1", "renderer_enable_thinking": True,
        "training_batch_size": 8, "owner_rows": len(owner_rows), "rehearsal_rows": len(rehearsal),
        "authored_handoffs_not_live_delegation": True, "incorrect_schema_action_masked": True,
        "wait_repair_episodes": sum(c["wait_repair"] for c in cases),
        "incorrect_wait_action_masked": include_wait_repairs,
        "wait_repair_context": ("authored_finite_two_poll_analogue_not_execution_of_observed_infinite_loop"
                                if include_wait_repairs else None),
        "runtime_context": {"path": str(runtime_trace.resolve()), "sha256": sha256_file(runtime_trace),
                            "usage": "runtime prefix and tool schema only; no evaluation sources or outputs"},
        "training_source": {"path": str(training_dir.resolve()),
                            "manifest_sha256": sha256_file(training_dir / "MANIFEST.json"),
                            "cases_sha256": sha256_file(training_dir / "CASES.json")},
        "rehearsal_source": {"path": str(rehearsal_dir.resolve()),
                             "manifest_sha256": sha256_file(rehearsal_dir / "MANIFEST.json"),
                             "parquet_sha256": sha256_file(rehearsal_dir / "train.parquet")},
        "cases_sha256": sha256_file(output_dir / "CASES.json"),
        "dataset": {"path": "train.parquet", "sha256": sha256_file(output_dir / "train.parquet")},
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-trace", "training-dir", "rehearsal-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--include-wait-repairs", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.runtime_trace, args.training_dir, args.rehearsal_dir, args.output_dir,
                            include_wait_repairs=args.include_wait_repairs), indent=2))
