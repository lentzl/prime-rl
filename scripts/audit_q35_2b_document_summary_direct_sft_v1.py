#!/usr/bin/env python3
"""Verify direct teacher file observations and real trainer token/loss boundaries."""

from __future__ import annotations

import argparse
import ast
import asyncio
import inspect
import io
import json
import re
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_document_summary_direct_sft_v1 import NATIVE_FAMILIES, NATIVE_REVISION_FAMILIES, SCHEMA_VERSION
from export_q35_2b_document_summary_evidence_sft_v1 import ROOT


async def _verify_native_file_observations(row, case, workspace):
    from document_summary_v1.taskset import MARKDOWN_CHILD_RECOVERY_FEEDBACK

    job = case["native_job"]
    revision = case["family"] in NATIVE_REVISION_FAMILIES
    masked = [4] if revision else ([6] if case["family"] == "native_count_repair" else [])
    if (case["masked_message_indices"] != masked
            or any(message.get("trainable") is not (index not in masked)
                   for index, message in enumerate(row["messages"]) if message["role"] == "assistant")):
        raise ValueError("native incorrect-context mask differs from its teaching boundary")
    if revision and (
            case.get("revision_provenance") != "authored_self_review_before_receipt_not_native_gate_feedback"
            or row["messages"][8].get("reasoning_content") != case["correction_reasoning"]
            or case["incorrect_draft"] == case["summary"]):
        raise ValueError("native revision provenance or reviewed target differs")
    if ("\nRecursive agent depth: 1\n" not in row["messages"][0]["content"]
            or row["messages"][1]["content"] != "[task from parent]\n\n" + job["prompt"]):
        raise ValueError("native child runtime or assignment differs from the declared job")
    chapter_id = job["chapter_id"]
    if (re.fullmatch(r"chapter-\d{3}", chapter_id) is None
            or job["worker"] != f"{chapter_id}-summarizer"
            or job["source_path"] != f"{ROOT}/chapters/{chapter_id}/source.md"
            or job["summary_path"] != f"{ROOT}/chapters/{chapter_id}/summary.md"
            or case["summary_word_count"] != sum(len(line[2:].split()) for line in case["summary"].splitlines())
            or not 0 < case["summary_word_count"] <= case["word_budget"]):
        raise ValueError("invalid native chapter identity, scoped paths or word-count target")
    def remap(text):
        return text.replace(ROOT, str(workspace))

    source_path, summary_path = Path(remap(job["source_path"])), Path(remap(job["summary_path"]))
    source_path.parent.mkdir(parents=True)
    source_path.write_text(case["source"], encoding="utf-8")
    sent, observations, writes = [], {}, 0

    async def send(message, *, receiver_role):
        expected = {"chapter_id": job["chapter_id"], "summary_path": str(summary_path)}
        if (sent or receiver_role != "parent" or json.loads(message) != expected
                or summary_path.read_text() != case["summary"]):
            raise ValueError("child must save the reviewed summary before one exact parent receipt")
        sent.append(message)
        return {"deliveryStatus": "queued"}

    scope = {"agent_message": SimpleNamespace(send=send)}
    for message in row["messages"]:
        for call in message.get("tool_calls") or []:
            if call["function"]["name"] != "ipython":
                raise ValueError("child episode replaced the native IPython interface")
            program = ast.parse(remap(json.loads(call["function"]["arguments"])["code"]))
            writes += sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                          and node.func.attr == "write_text" for node in ast.walk(program))
            output = io.StringIO()
            with redirect_stdout(output):
                execution = eval(compile(program, "child-teacher", "exec",
                                         flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), scope)
                if inspect.isawaitable(execution):
                    await execution
            observed = output.getvalue().replace(str(workspace), ROOT)
            if call["id"] == "repeat-write":
                observed = observed.rstrip() + "\n\n" + MARKDOWN_CHILD_RECOVERY_FEEDBACK
            observations[call["id"]] = observed
            if call["id"] == "write-summary-draft" and summary_path.read_text() != case["incorrect_draft"]:
                raise ValueError("native draft differs from the declared incorrect context")
        if message["role"] == "tool" and observations[message["tool_call_id"]] != message["content"]:
            raise ValueError("native child scripted observation differs from execution")
    expected_writes = 2 if case["family"] in NATIVE_REVISION_FAMILIES | {"native_count_repair"} else 1
    if (len(sent) != 1 or writes != expected_writes or summary_path.read_text() != case["summary"]
            or source_path.read_text() != case["source"]
            or set(p for p in workspace.rglob('*') if p.is_file()) != {source_path, summary_path}):
        raise ValueError("child did not preserve its source, save once, report once and stop")


def verify_file_observations(row, case, workspace: Path):
    if case["family"] in NATIVE_FAMILIES:
        return asyncio.run(_verify_native_file_observations(row, case, workspace))
    workspace.mkdir()
    (workspace / "source.md").write_text(case["source"], encoding="utf-8")
    scope, observed = {}, {}
    for message in row["messages"]:
        for call in message.get("tool_calls") or []:
            code = json.loads(call["function"]["arguments"])["code"].replace(ROOT, str(workspace))
            program = ast.parse(code)
            if not isinstance(program.body[-1], ast.Expr):
                raise ValueError("teacher tool call must expose its actual result")
            exec(compile(ast.Module(body=program.body[:-1], type_ignores=[]), "teacher", "exec"), scope)
            value = eval(compile(ast.Expression(program.body[-1].value), "teacher", "eval"), scope)
            observed[call["id"]] = repr(value)
            if case["family"] == "semantic_repair" and call["id"] == "write-summary-draft":
                if (workspace / "summary.md").read_text() != case["incorrect_draft"]:
                    raise ValueError("semantic draft differs from reviewed incorrect context")
        if message["role"] == "tool" and observed[message["tool_call_id"]] != message["content"]:
            raise ValueError("teacher observation does not match execution")
    if (workspace / "summary.md").read_text() != case["summary"]:
        raise ValueError("teacher does not write the reviewed summary")
    if (workspace / "source.md").read_text() != case["source"] or (workspace / "notes.md").exists():
        raise ValueError("teacher changed the source or introduced a notes stage")


def audit(dataset_dir: Path, tokenizer_path: Path):
    import torch
    from audit_q35_2b_document_summary_commit_revision_renderer_v3 import _renderer_messages, _renderer_tools
    from renderers.configs import Qwen35RendererConfig
    from renderers.qwen35 import Qwen35Renderer
    from transformers import AutoTokenizer

    from prime_rl.trainer.sft.data import SFTDataset

    manifest = json.loads((dataset_dir / "MANIFEST.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected dataset schema")
    cases = json.loads((dataset_dir / "CASES.json").read_text())
    if (
        sha256_file(dataset_dir / "train.parquet") != manifest["dataset"]["sha256"]
        or sha256_file(dataset_dir / "CASES.json") != manifest["cases_sha256"]
    ):
        raise ValueError("dataset content changed")
    rows = Dataset.from_parquet(str(dataset_dir / "train.parquet"))
    if len(rows) != len(cases) or len(rows) != manifest["rows"] or len({c["slug"] for c in cases}) != len(rows):
        raise ValueError("episode count or identity differs from manifest")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    renderer = Qwen35Renderer(tokenizer, Qwen35RendererConfig(enable_thinking=True))
    records = []
    with tempfile.TemporaryDirectory(prefix="direct-summary-replay-") as temporary:
        for row, case in zip(rows, cases, strict=True):
            if row["task_key"] != f"summary-direct-{case['slug']}":
                raise ValueError("case order differs from training rows")
            verify_file_observations(row, case, Path(temporary) / case["slug"])
            messages, tools = _renderer_messages(row["messages"]), _renderer_tools(row["tools"])
            repair = case["family"] in {"format_repair", "semantic_repair"}
            native = case["family"] in NATIVE_FAMILIES
            count_repair = case["family"] == "native_count_repair"
            native_revision = case["family"] in NATIVE_REVISION_FAMILIES
            roles = ["user", "user", "assistant", "tool"]
            if repair:
                roles += ["assistant", "tool", "assistant", "user"]
            roles += ["assistant", "tool", "assistant"]
            if native:
                roles = ["user", "user"] + ["assistant", "tool"] * (5 if count_repair or native_revision else 3) + ["assistant"]
            if [m["role"] for m in messages] != roles:
                raise ValueError("not the declared read/write/report/stop episode")
            if ((case["family"] == "semantic_repair" or native_revision)
                    and row["messages"][8]["reasoning_content"] != case["correction_reasoning"]):
                raise ValueError("semantic correction reasoning differs from reviewed target")
            masked_prefix = {4} if native_revision else ({6} if count_repair else ({4, 6} if repair else set()))
            if native and case["masked_message_indices"] != sorted(masked_prefix):
                raise ValueError("native retry mask annotation differs from its episode")
            if any(
                m.get("trainable") is not (i not in masked_prefix)
                for i, m in enumerate(row["messages"]) if m["role"] == "assistant"
            ):
                raise ValueError("incorrect draft/stop masking or missing positive action supervision")
            full = renderer.render(messages, tools=tools)
            ids = list(full.token_ids)
            sample = next(iter(SFTDataset(Dataset.from_list([row]), renderer, shuffle=False, seq_len=16384)))
            if sample["input_ids"] != ids[:-1] or sample["target_ids"] != ids[1:] or len(ids) > 16384:
                raise ValueError(f"truncated or changed training sequence: {case['slug']}: {len(ids)}")
            expected = [
                bool(mask) and index not in masked_prefix
                for mask, index in zip(full.sampled_mask[1:], full.message_indices[1:], strict=True)
            ]
            masked_prefix_tokens = sum(
                bool(mask) and index in masked_prefix
                for mask, index in zip(full.sampled_mask[1:], full.message_indices[1:], strict=True)
            )
            if (repair or count_repair or native_revision) and masked_prefix_tokens == 0:
                raise ValueError("incorrect draft context has no renderer-sampled tokens to mask")
            if sample["loss_mask"] != expected or not any(expected):
                raise ValueError("loss mask differs from renderer-sampled assistant tokens")
            if any(
                mask and messages[index]["role"] != "assistant"
                for mask, index in zip(sample["loss_mask"], full.message_indices[1:], strict=True)
            ):
                raise ValueError("source or task text contributes to loss")
            if any(mask and index in masked_prefix for mask, index in
                   zip(sample["loss_mask"], full.message_indices[1:], strict=True)):
                raise ValueError("incorrect draft or premature stop contributes to loss")
            supervised = [token for token, mask in zip(sample["target_ids"], expected, strict=True) if mask]
            decoded = tokenizer.decode(supervised)
            if "Done." not in decoded or "summary_text" not in decoded or "read_text" not in decoded:
                raise ValueError("training omits one of read/write/stop actions")
            if native and ("agent_message.send" not in decoded or "receiver_role" not in decoded):
                raise ValueError("native child training omits its parent receipt")
            if native_revision and case["correction_reasoning"] not in decoded:
                raise ValueError("native self-review correction reasoning is absent from supervised tokens")
            records.append(
                {
                    "slug": case["slug"],
                    "tokens": len(ids),
                    "supervised_tokens": len(supervised),
                    "file_observations_reproduced": True,
                    "truncated": False,
                    "source_or_user_supervised_tokens": 0,
                    "incorrect_prefix_supervised_tokens": 0,
                    "incorrect_prefix_context_tokens": masked_prefix_tokens,
                    "format_repair": case["family"] == "format_repair",
                    "semantic_repair": case["family"] == "semantic_repair",
                    "native_child": native,
                    "native_count_repair": count_repair,
                    "native_format_repair": case["family"] == "native_format_repair",
                    "native_semantic_repair": case["family"] == "native_semantic_repair",
                    "native_revision_reasoning_supervised": native_revision,
                    "receipt_send_stub_only": native,
                    "native_child_execution_verified": False,
                }
            )
    result = {
        "schema_version": "qwen35-2b-document-summary-direct-renderer-audit/v1",
        "status": "complete",
        "dataset_schema_version": SCHEMA_VERSION,
        "dataset_manifest_sha256": sha256_file(dataset_dir / "MANIFEST.json"),
        "dataset_parquet_sha256": sha256_file(dataset_dir / "train.parquet"),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
        "selected_enable_thinking": True,
        "seq_len": 16384,
        "rows": records,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    if result["cuda_initialized"]:
        raise ValueError("renderer audit unexpectedly initialized CUDA")
    output = dataset_dir / "RENDERER-AUDIT.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return {
        "rows": len(records),
        "max_tokens": max(r["tokens"] for r in records),
        "tokens": sum(r["tokens"] for r in records),
        "supervised_tokens": sum(r["supervised_tokens"] for r in records),
        "all_checks_pass": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset_dir, args.tokenizer), indent=2))
