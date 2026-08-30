#!/usr/bin/env python3
"""Build an answer-free SFT bootstrap for the document coordinator's first action."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

SCHEMA_VERSION = "qwen35-2b-document-decision-sft/v2"
OBJECTIVE = "canonical_answer_free_first_document_action"
FAMILIES = {"document_direct", "document_flat", "document_hierarchical"}
STEMS = ("alpha", "beta", "gamma")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wire_message(message: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in message.items() if value is not None}
    content = result.get("content")
    if isinstance(content, list):
        if not all(
            isinstance(part, dict) and part.get("type") == "text" for part in content
        ):
            raise ValueError("document decision SFT supports text-only messages")
        result["content"] = "".join(str(part.get("text", "")) for part in content)
    return result


def _worker_instruction(path: str) -> str:
    return (
        f"Read {path} using the CLI or IPython. Count words with Python str.split() over the "
        "complete file contents and count lines beginning exactly `## `. Send one JSON object "
        "with integer keys `words` and `h2` to your parent using "
        "await agent_message.send(json.dumps(result), receiver_role='parent'). After the "
        "delivery receipt succeeds, stop."
    )


def _manager_instruction(root: str) -> str:
    assignments = "\n".join(
        f"- {stem}-document-worker owns {root}/{stem}.md" for stem in STEMS
    )
    final_keys = ", ".join(
        [
            *(f"{stem}_words, {stem}_h2" for stem in STEMS),
            "total_words, total_h2",
        ]
    )
    return (
        "[recursive document coordinator session contract]\n"
        "session_role=document_coordinator\n"
        "is_root=false\n"
        "has_parent=true\n"
        "can_delegate=true\n"
        "can_finalize_user=false\n"
        "maximum_descendant_depth=1\n"
        "return_contract=exactly_one_parent_report\n"
        f"You own the document directory {root}. Do not compute the three file statistics "
        "yourself. Delegate all three assignments below to three independent terminal children, "
        "retaining their handles and spawning them before waiting:\n"
        f"{assignments}\n"
        "Each child must read only its assigned file, count all words with Python str.split(), "
        "count lines beginning exactly `## `, and send one JSON object with integer keys "
        "`words` and `h2` to you through agent_message.send. After all three explicit child "
        "reports arrive, assemble one JSON object with the per-file values and totals. Its exact "
        f"keys are: {final_keys}. Send that object exactly once to receiver_role='parent', then stop."
    )


def canonical_first_action(family: str, root: str) -> tuple[str, str]:
    """Return answer-free reasoning and IPython code for one topology."""
    if family == "document_direct":
        reasoning = (
            "This is root-local document work. I will inventory and read every Markdown file "
            "with Python, then compute the requested counts from the contents."
        )
        code = f"""from pathlib import Path
root = Path({root!r})
paths = sorted(root.glob('*.md'))
docs = {{path.stem: path.read_text() for path in paths}}
stats = {{
    stem: {{
        'words': len(text.split()),
        'h2': sum(line.startswith('## ') for line in text.splitlines()),
    }}
    for stem, text in docs.items()
}}
result = {{
    **{{f'{{stem}}_words': stats[stem]['words'] for stem in ('alpha', 'beta', 'gamma')}},
    **{{f'{{stem}}_h2': stats[stem]['h2'] for stem in ('alpha', 'beta', 'gamma')}},
    'total_words': sum(stats[stem]['words'] for stem in ('alpha', 'beta', 'gamma')),
    'total_h2': sum(stats[stem]['h2'] for stem in ('alpha', 'beta', 'gamma')),
}}
result"""
        return reasoning, code
    if family == "document_flat":
        reasoning = (
            "The root must delegate these independent files. I will admit all three named "
            "terminal workers now, retain their handles, and then yield for explicit reports."
        )
        calls = [
            f"{stem}_worker = await rlm({_worker_instruction(f'{root}/{stem}.md')!r}, "
            f"name='{stem}-document-worker')"
            for stem in STEMS
        ]
        return reasoning, "\n".join(calls)
    if family == "document_hierarchical":
        reasoning = (
            "The root must route this directory through one non-root coordinator. I will admit "
            "the named manager with the complete recursive contract and retain its handle."
        )
        code = (
            f"document_manager = await rlm({_manager_instruction(root)!r}, "
            "name='document-manager')"
        )
        return reasoning, code
    raise ValueError(f"unsupported document family: {family}")


def _canonical_row(trace: dict[str, Any], *, source: Path) -> dict[str, Any]:
    task = trace.get("task", {}).get("data", {})
    family = task.get("family")
    if family not in FAMILIES:
        raise ValueError(f"trace is not a document decision task: {family!r}")
    name = task.get("name")
    files = task.get("files")
    if not isinstance(name, str) or not isinstance(files, dict) or len(files) != 3:
        raise ValueError("document trace lacks a stable task name or three fixtures")
    roots = {str(Path(path).parent) for path in files}
    if len(roots) != 1:
        raise ValueError("document fixtures do not share one directory")
    root = roots.pop()
    expected_paths = {f"{root}/{stem}.md" for stem in STEMS}
    if set(files) != expected_paths:
        raise ValueError("document trace fixtures do not match the canonical three files")

    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 3:
        raise ValueError("document trace has no first sampled response")
    if nodes[0].get("sampled") is not False or nodes[1].get("sampled") is not False:
        raise ValueError("document trace prefix is not the canonical two-message prefix")
    first_sampled = next(
        (node for node in nodes if node.get("sampled") is True),
        None,
    )
    if first_sampled is None or first_sampled.get("message", {}).get("role") != "assistant":
        raise ValueError("document trace has no sampled assistant response")
    tools = trace.get("tools") or []
    if [tool.get("name") for tool in tools if isinstance(tool, dict)] != ["ipython"]:
        raise ValueError("document trace must expose exactly the IPython tool")

    reasoning, code = canonical_first_action(family, root)
    digest = hashlib.sha256(f"{trace.get('id')}:{family}:{code}".encode()).hexdigest()[:16]
    target = {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning,
        "tool_calls": [
            {
                "id": f"document-decision-{digest}",
                "type": "function",
                "function": {
                    "name": "ipython",
                    "arguments": json.dumps({"code": code}, separators=(",", ":")),
                },
            }
        ],
    }
    return {
        "messages": [
            _wire_message(nodes[0]["message"]),
            _wire_message(nodes[1]["message"]),
            target,
        ],
        "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
        "task_key": name,
        "trace_id": f"document-decision:{trace.get('id')}",
        "family": family,
        "role": "coordinator",
        "objective": OBJECTIVE,
        "source_trace": str(source),
    }


def export(*, traces: list[Path], output_dir: Path, expected_rows: int = 12) -> dict[str, Any]:
    if expected_rows != 12:
        raise ValueError("the v1 document decision bootstrap is fixed at twelve rows")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite document decision bootstrap: {output_dir}")
    candidates: dict[str, dict[str, Any]] = {}
    source_records = []
    for path in traces:
        if not path.is_file():
            raise FileNotFoundError(path)
        source_records.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                envelope = json.loads(line)
                for trace in envelope.get("traces") or []:
                    row = _canonical_row(trace, source=path.resolve())
                    if row["task_key"] in candidates:
                        raise ValueError(f"duplicate document task: {row['task_key']}")
                    candidates[row["task_key"]] = row
    rows = sorted(candidates.values(), key=lambda row: row["task_key"])
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} document rows, found {len(rows)}")
    family_counts = {
        family: sum(row["family"] == family for row in rows) for family in sorted(FAMILIES)
    }
    if set(family_counts.values()) != {4}:
        raise ValueError(f"document bootstrap is not topology balanced: {family_counts}")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "rows": len(rows),
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": source_records,
        "answer_free": True,
        "tool_call_format": "openai_function_v1",
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=12)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                traces=[path.resolve() for path in args.traces],
                output_dir=args.output_dir.resolve(),
                expected_rows=args.expected_rows,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
