from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = REPO_ROOT / "scripts/build_prime_agent_designer_document_corpus_v1.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numbered_markdown_heading_is_extracted_by_semantic_title() -> None:
    module = _module()
    markdown = """# Guide

### 2. Subagents are native RLM calls

Admission returns a handle, not an answer.

#### Child lifecycle

Keep the handle until the reply arrives.

### 3. Skills

Unrelated.
"""

    assert module._section(markdown, "Subagents are native RLM calls") == (
        "Admission returns a handle, not an answer.\n\n"
        "#### Child lifecycle\n\nKeep the handle until the reply arrives."
    )


def test_document_corpus_build_is_pinned_and_reproducible(tmp_path, monkeypatch) -> None:
    module = _module()
    bodies = {}
    for item in module.SECTIONS:
        bodies.setdefault(item["path"], "# Document\n")
        bodies[item["path"]] += f"\n## {item['heading']}\n\nProtocol text for {item['heading']}.\n"
    monkeypatch.setattr(module, "_fetch", bodies.__getitem__)
    output = tmp_path / "corpus.json"

    first = module.build(output=output)
    second = module.build(output=output)

    assert first == second == json.loads(output.read_text())
    assert first["schema_version"] == module.SCHEMA_VERSION
    assert first["upstream"]["revision"] == module.UPSTREAM_REVISION
    assert len(first["documents"]) == len(module.SECTIONS)
    assert all(document["content_sha256"] for document in first["documents"])
