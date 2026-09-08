#!/usr/bin/env python3
"""Verify direct teacher file observations and real trainer token/loss boundaries."""

from __future__ import annotations

import argparse
import ast
import json
import tempfile
from pathlib import Path

import torch
from audit_q35_2b_document_summary_commit_revision_renderer_v3 import _renderer_messages, _renderer_tools
from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_document_summary_direct_sft_v1 import SCHEMA_VERSION
from export_q35_2b_document_summary_evidence_sft_v1 import ROOT
from renderers.configs import Qwen35RendererConfig
from renderers.qwen35 import Qwen35Renderer
from transformers import AutoTokenizer

from prime_rl.trainer.sft.data import SFTDataset


def verify_file_observations(row, case, workspace: Path):
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
        if message["role"] == "tool" and observed[message["tool_call_id"]] != message["content"]:
            raise ValueError("teacher observation does not match execution")
    if (workspace / "summary.md").read_text() != case["summary"]:
        raise ValueError("teacher does not write the reviewed summary")
    if (workspace / "source.md").read_text() != case["source"] or (workspace / "notes.md").exists():
        raise ValueError("teacher changed the source or introduced a notes stage")


def audit(dataset_dir: Path, tokenizer_path: Path):
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
    if len(rows) != len(cases) or len(rows) != 40:
        raise ValueError("expected forty complete teacher episodes")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    renderer = Qwen35Renderer(tokenizer, Qwen35RendererConfig(enable_thinking=True))
    records = []
    with tempfile.TemporaryDirectory(prefix="direct-summary-replay-") as temporary:
        for row, case in zip(rows, cases, strict=True):
            if row["task_key"] != f"summary-direct-{case['slug']}":
                raise ValueError("case order differs from training rows")
            verify_file_observations(row, case, Path(temporary) / case["slug"])
            messages, tools = _renderer_messages(row["messages"]), _renderer_tools(row["tools"])
            if [m["role"] for m in messages] != ["user", "user", "assistant", "tool", "assistant", "tool", "assistant"]:
                raise ValueError("not a direct read/write/stop episode")
            full = renderer.render(messages, tools=tools)
            ids = list(full.token_ids)
            sample = next(iter(SFTDataset(Dataset.from_list([row]), renderer, shuffle=False, seq_len=16384)))
            if sample["input_ids"] != ids[:-1] or sample["target_ids"] != ids[1:] or len(ids) > 16384:
                raise ValueError(f"truncated or changed training sequence: {case['slug']}: {len(ids)}")
            expected = [bool(x) for x in full.sampled_mask[1:]]
            if sample["loss_mask"] != expected or not any(expected):
                raise ValueError("loss mask differs from renderer-sampled assistant tokens")
            if any(
                mask and messages[index]["role"] != "assistant"
                for mask, index in zip(sample["loss_mask"], full.message_indices[1:], strict=True)
            ):
                raise ValueError("source or task text contributes to loss")
            supervised = [token for token, mask in zip(sample["target_ids"], expected, strict=True) if mask]
            decoded = tokenizer.decode(supervised)
            if "Done." not in decoded or "summary_text" not in decoded or "read_text" not in decoded:
                raise ValueError("training omits one of read/write/stop actions")
            records.append(
                {
                    "slug": case["slug"],
                    "tokens": len(ids),
                    "supervised_tokens": len(supervised),
                    "file_observations_reproduced": True,
                    "truncated": False,
                    "source_or_user_supervised_tokens": 0,
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
