#!/usr/bin/env python3
"""Replay scripted owner operations and audit real trainer token/loss boundaries."""

from __future__ import annotations

import argparse
import ast
import asyncio
import inspect
import io
import json
import tempfile
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_document_summary_owner_sft_v1 import (
    INDEX_PATH,
    MARKDOWN_OUTPUT_PATH,
    OBJECTIVE,
    SCHEMA_VERSION,
)


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def validate_dataset(path):
    manifest = json.loads((path / "MANIFEST.json").read_text())
    cases = json.loads((path / "CASES.json").read_text())
    owner_counts = Counter(c["family"] for c in cases)
    wait_count = owner_counts.get("owner_wait_repair", 0)
    start_count = owner_counts.get("owner_start_repair", 0)
    if (not cases or set(owner_counts) - {
            "owner_delegation_fanin", "owner_schema_receipt_repair", "owner_wait_repair", "owner_start_repair"}
            or (start_count and (
                manifest.get("start_repair_episodes") != start_count
                or manifest.get("incorrect_start_response_masked") is not True
                or manifest.get("start_repair_context") !=
                "authored_short_repetition_or_premature_wait_with_no_executed_action"
                or manifest.get("start_repair_feedback") != "authored_state_correction_not_native_gate_feedback"))
            or (wait_count and (
                manifest.get("wait_repair_episodes") != wait_count
                or manifest.get("incorrect_wait_action_masked") is not True
                or manifest.get("wait_repair_context") !=
                "authored_finite_two_poll_analogue_not_execution_of_observed_infinite_loop"))):
        raise ValueError("invalid owner episode families or waiting-repair provenance")
    if (manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete"
            or manifest.get("role") != "coordinator" or manifest.get("objective") != OBJECTIVE
            or manifest.get("rows") != len(cases) + 48 or manifest.get("owner_rows") != len(cases)
            or manifest.get("rehearsal_rows") != 48 or manifest.get("answer_free") is not False
            or manifest.get("renderer_enable_thinking") is not True
            or manifest.get("authored_handoffs_not_live_delegation") is not True
            or manifest.get("incorrect_schema_action_masked") is not True
            or manifest.get("tool_call_format") != "openai_function_v1"
            or manifest.get("dataset", {}).get("path") != "train.parquet"
            or sha256_file(path / "train.parquet") != manifest["dataset"]["sha256"]
            or sha256_file(path / "CASES.json") != manifest["cases_sha256"]
            or len({c["task_key"] for c in cases}) != len(cases)):
        raise ValueError("invalid scripted owner training dataset")
    rows = list(Dataset.from_parquet(str(path / "train.parquet")))
    expected = {**owner_counts,
                "adaptive_solve_owned": 16, "adaptive_delegate_terminal": 16,
                "adaptive_delegate_coordinator": 16}
    if (len(rows) != manifest["rows"] or len({r["task_key"] for r in rows}) != len(rows)
            or dict(Counter(r["family"] for r in rows)) != expected
            or manifest.get("family_counts") != expected
            or manifest.get("task_keys") != [r["task_key"] for r in rows]):
        raise ValueError("owner/rehearsal mixture identity mismatch")
    by_key = {c["task_key"]: c for c in cases}
    if {r["task_key"] for r in rows if r["family"] in owner_counts} != set(by_key):
        raise ValueError("owner case/row identity mismatch")
    for row in rows:
        case = by_key.get(row["task_key"])
        if case is not None and row["family"] != case["family"]:
            raise ValueError("owner case/row family mismatch")
        verify_masks(row, case)
    return manifest, rows, by_key


def verify_masks(row, case):
    expected = set()
    if case is not None:
        schema_repair = case["schema_repair"]
        wait_repair = case.get("wait_repair", False)
        start_repair = case.get("start_repair")
        family = ("owner_start_repair" if start_repair else "owner_wait_repair" if wait_repair else
                  "owner_schema_receipt_repair" if schema_repair else "owner_delegation_fanin")
        if case["family"] != family or sum(map(bool, (wait_repair, schema_repair, start_repair))) > 1:
            raise ValueError("inconsistent owner repair family")
        if start_repair:
            if start_repair not in ("repetition", "premature_wait"):
                raise ValueError("invalid start-repair boundary")
            expected = {2}
        if schema_repair:
            expected = {4}
        if wait_repair:
            stage = case.get("wait_repair_after_receipts")
            if stage not in (0, 1) or stage >= len(case["chapters"]):
                raise ValueError("invalid waiting-repair boundary")
            expected = {6 if stage == 0 else 10}
    masked = {i for i, message in enumerate(_clean(row["messages"]))
              if message.get("trainable") is False}
    if masked != expected:
        raise ValueError("wrong owner incorrect-action mask")
    return masked


async def verify_episode(row, case, workspace):
    """Replay file I/O with a declared stub for native admission, not live children."""
    workspace.mkdir()
    root = "/workspace/document-summary-v1"
    local_root = str(workspace / "document")
    local_output = str(workspace / "artifacts/summary.md")

    def remap(text):
        return text.replace(MARKDOWN_OUTPUT_PATH, local_output).replace(root, local_root)

    def restore(text):
        return text.replace(local_output, MARKDOWN_OUTPUT_PATH).replace(local_root, root)

    index = json.loads(remap(json.dumps(case["index"])))
    index_path = Path(remap(INDEX_PATH))
    index_path.parent.mkdir()
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    original_index = index_path.read_bytes()
    Path(local_output).parent.mkdir()
    jobs = {job["worker"]: job for job in index["chapters"]}
    admitted, received, observations = {}, set(), {}
    next_delivery = 0

    async def admit(prompt, *, name):
        if name not in jobs or name in admitted or prompt != jobs[name]["prompt"]:
            raise ValueError("duplicate worker or changed native name/prompt")
        handle = object()
        admitted[name] = handle
        return handle

    scope = {"rlm": admit}
    for message in _clean(row["messages"]):
        if message["role"] == "user" and message["content"].startswith("[from child:"):
            if set(admitted) != set(jobs):
                raise ValueError("scripted receipt before all admissions")
            delivery = case["deliveries"][next_delivery]
            next_delivery += 1
            worker, payload = delivery["worker"], delivery["payload"]
            if (not message["content"].startswith(f"[from child:{worker}]\n")
                    or json.loads(message["content"].rsplit("\n\n", 1)[1]) != payload):
                raise ValueError("scripted receipt differs from declared delivery")
            if delivery["valid"]:
                received.add(worker)
                summary_path = Path(jobs[worker]["summary_path"])
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(case["summaries"][worker], encoding="utf-8")
        for call in message.get("tool_calls", []):
            if call["function"]["name"] != "ipython":
                raise ValueError("owner episode replaced the native IPython interface")
            code = remap(json.loads(call["function"]["arguments"])["code"])
            program = ast.parse(code)
            output = io.StringIO()
            with redirect_stdout(output):
                execution = eval(compile(program, "owner-teacher", "exec",
                                         flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), scope)
                if inspect.isawaitable(execution):
                    await execution
            observations[call["id"]] = restore(output.getvalue())
            if received != set(jobs) and Path(local_output).exists():
                raise ValueError("owner assembled before all matching receipts")
        if message["role"] == "tool" and observations[message["tool_call_id"]] != message["content"]:
            raise ValueError(f"scripted tool observation mismatch: {message['tool_call_id']}")
    expected = "\n\n".join(job["heading"] + "\n\n" + case["summaries"][job["worker"]].strip()
                            for job in index["chapters"]) + "\n"
    if (set(admitted) != set(jobs) or next_delivery != len(case["deliveries"])
            or scope["handles"] != admitted or scope["receipts"] != {
                name: {"chapter_id": job["chapter_id"], "summary_path": job["summary_path"]}
                for name, job in jobs.items()}
            or Path(local_output).read_text() != expected or index_path.read_bytes() != original_index):
        raise ValueError("owner did not retain handles, preserve index, match receipts or assemble unchanged")
    if any(Path(job["source_path"]).exists() for job in jobs.values()):
        raise ValueError("owner created a source file")
    return {"scripted_file_observations_reproduced": True, "admission_stub_only": True,
            "native_child_execution_verified": False, "chapter_count": len(jobs)}


def audit(dataset_dir, tokenizer_path, rehearsal_dir):
    import torch
    from audit_q35_2b_document_summary_commit_revision_renderer_v3 import _renderer_messages, _renderer_tools
    from renderers.configs import Qwen35RendererConfig
    from renderers.qwen35 import Qwen35Renderer
    from transformers import AutoTokenizer

    from prime_rl.trainer.sft.data import SFTDataset

    manifest, rows, cases = validate_dataset(dataset_dir)
    if sha256_file(rehearsal_dir / "train.parquet") != manifest["rehearsal_source"]["parquet_sha256"]:
        raise ValueError("changed foundational rehearsal")
    old = {r["task_key"]: r for r in Dataset.from_parquet(str(rehearsal_dir / "train.parquet"))}
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    renderer = Qwen35Renderer(tokenizer, Qwen35RendererConfig(enable_thinking=True))
    records = []
    with tempfile.TemporaryDirectory(prefix="summary-owner-replay-") as temporary:
        for i, row in enumerate(rows):
            case = cases.get(row["task_key"])
            record = {"task_key": row["task_key"], "family": row["family"]}
            if case is not None:
                record.update(asyncio.run(verify_episode(row, case, Path(temporary) / str(i))))
            elif any(_clean(row[key]) != _clean(value) for key, value in old[row["task_key"]].items()):
                raise ValueError("acquired role rehearsal content changed")
            masked = verify_masks(row, case)
            messages = _renderer_messages(row["messages"])
            full = renderer.render(messages, tools=_renderer_tools(row["tools"]))
            ids = list(full.token_ids)
            sample = next(iter(SFTDataset(Dataset.from_list([row]), renderer, shuffle=False, seq_len=16384)))
            expected = [bool(mask) and index not in masked
                        for mask, index in zip(full.sampled_mask[1:], full.message_indices[1:], strict=True)]
            if (len(ids) > 16384 or sample["input_ids"] != ids[:-1] or sample["target_ids"] != ids[1:]
                    or sample["loss_mask"] != expected or not any(expected)
                    or any(mask and messages[index]["role"] != "assistant"
                           for mask, index in zip(expected, full.message_indices[1:], strict=True))):
                raise ValueError("owner/rehearsal sequence truncated or loss boundary changed")
            supervised_messages = {index for mask, index in
                                   zip(expected, full.message_indices[1:], strict=True) if mask}
            if supervised_messages != {i for i, message in enumerate(messages)
                                       if message["role"] == "assistant" and i not in masked}:
                raise ValueError("one or more positive owner/rehearsal actions receive no supervision")
            bad_tokens = sum(bool(mask) and index in masked
                             for mask, index in zip(full.sampled_mask[1:], full.message_indices[1:], strict=True))
            if masked and not bad_tokens:
                raise ValueError("incorrect owner action absent from masked context")
            record.update(tokens=len(ids), supervised_tokens=sum(expected), truncated=False,
                          source_or_user_supervised_tokens=0, incorrect_prefix_supervised_tokens=0,
                          incorrect_prefix_context_tokens=bad_tokens,
                          rehearsal_preserved=case is None)
            records.append(record)
    result = {
        "schema_version": "qwen35-2b-document-summary-owner-renderer-audit/v1", "status": "complete",
        "dataset_schema_version": SCHEMA_VERSION,
        "dataset_manifest_sha256": sha256_file(dataset_dir / "MANIFEST.json"),
        "dataset_parquet_sha256": sha256_file(dataset_dir / "train.parquet"),
        "tokenizer_path": str(tokenizer_path), "tokenizer_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
        "selected_enable_thinking": True, "seq_len": 16384, "rows": records,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    if result["cuda_initialized"]:
        raise ValueError("owner audit unexpectedly initialized CUDA")
    output = dataset_dir / "RENDERER-AUDIT.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return {"rows": len(records), "sequence_tokens": sum(r["tokens"] for r in records),
            "maximum_sequence_tokens": max(r["tokens"] for r in records),
            "supervised_tokens": sum(r["supervised_tokens"] for r in records),
            "masked_incorrect_prefix_tokens": sum(r["incorrect_prefix_context_tokens"] for r in records),
            "cuda_initialized": result["cuda_initialized"], "sha256": sha256_file(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-dir", "tokenizer-path", "rehearsal-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset_dir, args.tokenizer_path, args.rehearsal_dir), indent=2))
