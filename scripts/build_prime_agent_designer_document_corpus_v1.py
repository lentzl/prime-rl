#!/usr/bin/env python3
"""Fetch a pinned, protocol-focused Prime Agent corpus for the SPADE designer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "prime-agent-designer-document-corpus/v1"
UPSTREAM_REPOSITORY = "https://github.com/PrimeIntellect-ai/prime-agent"
UPSTREAM_REVISION = "9e49b73dd46908b3e400f4780b46a90daef69052"
RAW_ROOT = (
    "https://raw.githubusercontent.com/PrimeIntellect-ai/prime-agent/"
    f"{UPSTREAM_REVISION}/"
)
SECTIONS = (
    {
        "path": "packages/coding-agent/docs/rlm.md",
        "heading": "Subagents are native RLM calls",
        "tags": ["delegation", "admission", "explicit_reply", "child_lifecycle"],
    },
    {
        "path": "packages/coding-agent/docs/rlm.md",
        "heading": "State is designed to outlive one turn",
        "tags": ["persistence", "compaction", "long_running"],
    },
    {
        "path": "packages/coding-agent/docs/rlm-runtime.md",
        "heading": "Independent Delegation",
        "tags": ["delegation", "parallelism", "passive_yield"],
    },
    {
        "path": "packages/coding-agent/docs/rlm-runtime.md",
        "heading": "Parent-Scoped Sub-Agent Registry",
        "tags": ["retained_handle", "recovery", "child_lifecycle"],
    },
    {
        "path": "packages/coding-agent/docs/long-running-agents.md",
        "heading": "Agent-to-Agent Communication",
        "tags": ["messaging", "delivery", "follow_up"],
    },
    {
        "path": "packages/coding-agent/docs/long-running-agents.md",
        "heading": "Compaction and Continuity",
        "tags": ["persistence", "continuity", "child_lifecycle"],
    },
    {
        "path": "packages/coding-agent/skills/agent-message/SKILL.md",
        "heading": "API",
        "tags": ["messaging", "family_scope", "delivery_receipt"],
    },
    {
        "path": "packages/coding-agent/skills/agent-message/SKILL.md",
        "heading": "Safety",
        "tags": ["messaging", "family_scope", "child_lifecycle"],
    },
    {
        "path": "packages/coding-agent/skills/agent-observe/SKILL.md",
        "heading": "API",
        "tags": ["observation", "family_scope", "read_only"],
    },
    {
        "path": "packages/coding-agent/skills/agent-observe/SKILL.md",
        "heading": "Safety",
        "tags": ["observation", "family_scope", "read_only"],
    },
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _section(markdown: str, heading: str) -> str:
    lines = markdown.splitlines()
    matched_level = None
    selected: list[str] = []
    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if matched_level is None:
            source_heading = (
                re.sub(r"^\d+\.\s*", "", match.group(2).strip()) if match else None
            )
            if source_heading == heading:
                matched_level = len(match.group(1))
            continue
        if match and len(match.group(1)) <= matched_level:
            break
        selected.append(line.rstrip())
    content = "\n".join(selected).strip()
    if not content:
        raise ValueError(f"upstream document lacks non-empty section: {heading}")
    return content


def _fetch(path: str) -> str:
    request = urllib.request.Request(
        RAW_ROOT + path,
        headers={"User-Agent": "prime-rl-spade-document-corpus-v1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def build(*, output: Path) -> dict[str, Any]:
    sources = {item["path"]: _fetch(item["path"]) for item in SECTIONS}
    documents = []
    for item in SECTIONS:
        content = _section(sources[item["path"]], item["heading"])
        content_sha256 = _sha256_text(content)
        slug = re.sub(r"[^a-z0-9]+", "-", item["heading"].lower()).strip("-")
        documents.append(
            {
                "document_id": f"prime-agent-{slug}-{content_sha256[:12]}",
                "source_path": item["path"],
                "source_url": RAW_ROOT + item["path"],
                "source_sha256": _sha256_text(sources[item["path"]]),
                "heading": item["heading"],
                "tags": item["tags"],
                "content": content,
                "content_sha256": content_sha256,
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
        },
        "selection_policy": {
            "scope": "Prime Agent recursive coordination and lifecycle protocol",
            "answer_bearing_task_data_allowed": False,
            "verbatim_sections": True,
        },
        "documents": documents,
    }
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") != text:
        raise ValueError(f"refusing to replace a different document corpus: {output}")
    output.write_text(text, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(output=args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
