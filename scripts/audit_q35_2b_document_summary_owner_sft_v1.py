#!/usr/bin/env python3
"""Replay scripted owner operations and audit real trainer token/loss boundaries."""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import inspect
import io
import json
import tempfile
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_document_summary_owner_sft_v1 import (
    INDEX_PATH,
    MARKDOWN_OUTPUT_PATH,
    OBJECTIVE,
    SCHEMA_VERSION,
    UNNECESSARY_FOLLOWUP,
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
    prefixes = manifest.get("owner_supervision_boundary", "full_episode") == "assistant_turn_prefix"
    if manifest.get("owner_supervision_boundary", "full_episode") not in {"full_episode", "assistant_turn_prefix"}:
        raise ValueError("unsupported owner supervision boundary")
    sequence_counts = Counter()
    for case in cases:
        sequence_counts[case["family"]] += (
            sum(m["role"] == "assistant" and m.get("trainable") is not False
                for m in case["teacher_messages"]) if prefixes else 1
        )
    owner_rows = sum(sequence_counts.values())
    wait_count = owner_counts.get("owner_wait_repair", 0)
    start_count = owner_counts.get("owner_start_repair", 0)
    handle_count = owner_counts.get("owner_handle_repair", 0)
    message_count = owner_counts.get("owner_message_repair", 0)
    if (not cases or set(owner_counts) - {
            "owner_delegation_fanin", "owner_schema_receipt_repair", "owner_wait_repair", "owner_start_repair",
            "owner_handle_repair", "owner_message_repair"}
            or (message_count and (
                manifest.get("message_repair_episodes") != message_count
                or manifest.get("incorrect_message_actions_masked") is not True
                or manifest.get("message_repair_context") !=
                "authored_partial_send_and_metadata_confusion_with_declared_messaging_stub"
                or manifest.get("message_repair_observations") !=
                "documented_deliveryStatus_and_message_projection_not_full_native_response"))
            or (handle_count and (
                manifest.get("handle_repair_episodes") != handle_count
                or manifest.get("incorrect_handle_action_masked") is not True
                or manifest.get("handle_repair_context") !=
                "authored_join_or_await_failure_with_declared_admission_stub"))
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
            or manifest.get("rows") != owner_rows + 48 or manifest.get("owner_rows") != owner_rows
            or (prefixes and manifest.get("owner_teacher_episodes") != len(cases))
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
    expected = {**sequence_counts,
                "adaptive_solve_owned": 16, "adaptive_delegate_terminal": 16,
                "adaptive_delegate_coordinator": 16}
    if (len(rows) != manifest["rows"] or len({r["task_key"] for r in rows}) != len(rows)
            or dict(Counter(r["family"] for r in rows)) != expected
            or manifest.get("family_counts") != expected
            or manifest.get("task_keys") != [r["task_key"] for r in rows]):
        raise ValueError("owner/rehearsal mixture identity mismatch")
    by_key = {c["task_key"]: c for c in cases}
    if {r.get("episode_key") or r["task_key"] for r in rows if r["family"] in owner_counts} != set(by_key):
        raise ValueError("owner case/row identity mismatch")
    if prefixes:
        expected_keys = {f"{case['task_key']}:decision-{i}" for case in cases
                         for i, message in enumerate(case["teacher_messages"])
                         if message["role"] == "assistant" and message.get("trainable") is not False}
        if {row["task_key"] for row in rows if row["family"] in owner_counts} != expected_keys:
            raise ValueError("missing or duplicated owner decision boundary")
    for row in rows:
        case = by_key.get(row.get("episode_key") or row["task_key"])
        if case is not None and row["family"] != case["family"]:
            raise ValueError("owner case/row family mismatch")
        verify_masks(row, case)
    return manifest, rows, by_key


def verify_masks(row, case):
    if case is not None and row.get("episode_key"):
        index = row.get("target_message_index")
        teacher = case["teacher_messages"]
        verify_masks({"messages": teacher}, case)
        if (not isinstance(index, int) or not 0 <= index < len(teacher)
                or teacher[index]["role"] != "assistant" or teacher[index].get("trainable") is False
                or row["task_key"] != f"{case['task_key']}:decision-{index}"):
            raise ValueError("invalid owner decision target")
        expected_messages = copy.deepcopy(teacher[:index + 1])
        for message in expected_messages[:-1]:
            if message["role"] == "assistant":
                message["trainable"] = False
        if _clean(row["messages"]) != _clean(expected_messages):
            raise ValueError("owner prefix differs from its teacher episode or target mask")
        return {i for i, message in enumerate(expected_messages[:-1]) if message["role"] == "assistant"}
    expected = set()
    if case is not None:
        schema_repair = case["schema_repair"]
        wait_repair = case.get("wait_repair", False)
        start_repair = case.get("start_repair")
        handle_repair = case.get("handle_repair")
        message_repair = case.get("message_repair")
        family = ("owner_message_repair" if message_repair else "owner_handle_repair" if handle_repair else "owner_start_repair" if start_repair else "owner_wait_repair" if wait_repair else
                  "owner_schema_receipt_repair" if schema_repair else "owner_delegation_fanin")
        if case["family"] != family or sum(map(bool, (wait_repair, schema_repair, start_repair, handle_repair, message_repair))) > 1:
            raise ValueError("inconsistent owner repair family")
        if message_repair:
            boundary = case.get("message_repair_boundary")
            if (boundary not in ("before_receipts", "last_receipt_unstored")
                    or case.get("outgoing_delivery_status") not in ("queued", "delivered")):
                raise ValueError("invalid message-repair boundary or outgoing delivery status")
            start = 6 if boundary == "before_receipts" else 8 + 4 * (len(case["chapters"]) - 1)
            expected = {start, start + 2}
            calls = [(i, call["id"]) for i, message in enumerate(_clean(row["messages"]))
                     for call in message.get("tool_calls", [])
                     if call["id"] in {"unnecessary-followups", "wrong-send-result", "inspect-message-state"}]
            if calls != [(start, "unnecessary-followups"), (start + 2, "wrong-send-result"),
                         (start + 4, "inspect-message-state")]:
                raise ValueError("message misuse and recovery are not at the declared boundary")
        if handle_repair:
            boundary = case.get("handle_repair_boundary")
            if handle_repair not in ("join", "await") or boundary not in ("before_receipts", "last_receipt_unstored"):
                raise ValueError("invalid handle-repair boundary")
            expected = {6 if boundary == "before_receipts" else 8 + 4 * (len(case["chapters"]) - 1)}
            calls = [(i, call) for i, message in enumerate(_clean(row["messages"]))
                     for call in message.get("tool_calls", []) if call["id"] == "wrong-handle"]
            if len(calls) != 1 or {i for i, _ in calls} != expected:
                raise ValueError("handle misuse is not at its declared boundary")
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
    expected_receipts = {name: {"chapter_id": job["chapter_id"], "summary_path": job["summary_path"]}
                         for name, job in jobs.items()}
    admitted, received, observations, sent = {}, set(), {}, {}
    next_delivery = 0

    async def admit(prompt, *, name):
        if name not in jobs or name in admitted or prompt != jobs[name]["prompt"]:
            raise ValueError("duplicate worker or changed native name/prompt")
        handle = object()
        admitted[name] = handle
        return handle

    async def send(message, *, receiver_role, receiver_name):
        if (not case.get("message_repair") or receiver_name not in admitted or receiver_name in sent
                or receiver_role != "child" or message != UNNECESSARY_FOLLOWUP
                or len(received) != (0 if case["message_repair_boundary"] == "before_receipts" else len(jobs))):
            raise ValueError("unexpected outgoing message or wrong failure boundary")
        result = {"deliveryStatus": case["outgoing_delivery_status"], "message": message}
        sent[receiver_name] = result
        return result

    scope = {"rlm": admit, "agent_message": SimpleNamespace(send=send), "agent_observe": SimpleNamespace()}
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
            if call["id"] == "inspect-message-state":
                stored = {delivery["worker"] for delivery in case["deliveries"][:max(0, next_delivery - 1)]
                          if delivery["valid"]}
                if (scope["results"] != sent or set(sent) != set(jobs) or scope["handles"] != admitted
                        or scope["receipts"] != {name: expected_receipts[name] for name in stored}):
                    raise ValueError("message repair lost earlier sends, handles or stored receipts")
            if Path(local_output).exists() and (received != set(jobs) or scope.get("receipts") != expected_receipts):
                raise ValueError("owner assembled before all matching receipts")
        if message["role"] == "tool" and observations[message["tool_call_id"]] != message["content"]:
            raise ValueError(f"scripted tool observation mismatch: {message['tool_call_id']}")
    expected = "\n\n".join(job["heading"] + "\n\n" + case["summaries"][job["worker"]].strip()
                            for job in index["chapters"]) + "\n"
    if (set(admitted) != set(jobs) or next_delivery != len(case["deliveries"])
            or set(sent) != (set(jobs) if case.get("message_repair") else set())
            or scope["handles"] != admitted or scope["receipts"] != expected_receipts
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
    replayed = {}
    with tempfile.TemporaryDirectory(prefix="summary-owner-replay-") as temporary:
        for i, row in enumerate(rows):
            case = cases.get(row.get("episode_key") or row["task_key"])
            record = {"task_key": row["task_key"], "family": row["family"]}
            if case is not None:
                if row.get("episode_key"):
                    if case["task_key"] not in replayed:
                        replayed[case["task_key"]] = asyncio.run(verify_episode(
                            {"messages": case["teacher_messages"]}, case, Path(temporary) / str(i)))
                    record.update(replayed[case["task_key"]], teacher_episode_replayed=True,
                                  decision_prefix_exact=True, target_message_index=row["target_message_index"])
                else:
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
            if row.get("episode_key"):
                target_index = row["target_message_index"]
                target_text = tokenizer.decode([token for token, index in
                                                zip(full.token_ids, full.message_indices, strict=True)
                                                if index == target_index])
                reasoning = messages[target_index].get("reasoning_content", "").strip()
                supervised_text = tokenizer.decode([token for token, mask in
                                                    zip(full.token_ids[1:], expected, strict=True) if mask])
                if (not reasoning or reasoning not in target_text
                        or reasoning not in supervised_text
                        or "<think>" not in target_text or "</think>" not in target_text):
                    raise ValueError("current owner decision reasoning is absent from rendered supervision")
                record["target_reasoning_present"] = True
                record["target_reasoning_supervised"] = True
            incorrect = ({i for i, message in enumerate(case["teacher_messages"][:len(messages)])
                          if message.get("trainable") is False} if row.get("episode_key") else masked)
            bad_tokens = sum(bool(mask) and index in incorrect
                             for mask, index in zip(full.sampled_mask[1:], full.message_indices[1:], strict=True))
            if incorrect and not bad_tokens:
                raise ValueError("incorrect owner action absent from masked context")
            record.update(tokens=len(ids), supervised_tokens=sum(expected), truncated=False,
                          source_or_user_supervised_tokens=0, incorrect_prefix_supervised_tokens=0,
                          incorrect_prefix_context_tokens=bad_tokens,
                          masked_history_context_tokens=sum(bool(mask) and index in masked
                              for mask, index in zip(full.sampled_mask[1:], full.message_indices[1:], strict=True)),
                          rehearsal_preserved=case is None)
            records.append(record)
    result = {
        "schema_version": "qwen35-2b-document-summary-owner-renderer-audit/v1", "status": "complete",
        "dataset_schema_version": SCHEMA_VERSION,
        "owner_supervision_boundary": manifest.get("owner_supervision_boundary", "full_episode"),
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
