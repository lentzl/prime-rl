#!/usr/bin/env python3
"""Generate, score, remember, and export causal SPADE environment proposals."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from datasets import Dataset
from procedural_harness_master_v1.interaction_curriculum import (
    COEVOLUTION_CONTEXT_HEADER,
)
from procedural_harness_master_v1.taskset import (
    ProceduralHarnessMasterConfig,
    ProceduralHarnessMasterTaskset,
)

GENERATION_SCHEMA_VERSION = "qwen35-2b-spade-coevolution-generation/v1"
SCORE_SCHEMA_VERSION = "qwen35-2b-spade-coevolution-score/v1"
DESIGNER_CORPUS_SCHEMA_VERSION = "qwen35-2b-spade-rewarded-designer-corpus/v1"
DESIGNER_REPAIR_CORPUS_SCHEMA_VERSION = "qwen35-2b-spade-designer-repair-corpus/v1"
BOOTSTRAP_SCHEMA_VERSION = "qwen35-2b-environment-bootstrap-context/v1"
SUMMARY_SCHEMA_VERSION = "qwen35-2b-interaction-curriculum-summary/v1"
MEMORY_SCHEMA_VERSION = "qwen35-2b-spade-environment-memory/v1"
DOCUMENT_CORPUS_SCHEMA_VERSION = "prime-agent-designer-document-corpus/v1"
SYSTEM_PROMPT = """You are the Environment Designer in a co-evolving Prime Agents curriculum.
Produce one answer-free environment overlay and one privileged hint for the requested interaction
skill. The overlay must preserve the original executable task and scoring mechanism, resource ownership, one
child delegation, passive coordinator yield, and the child's use of IPython plus agent_message.send.
Make the no-hint environment challenging but feasible. The hint should expose a useful strategy,
never a private evidence value or final answer. Return only the requested JSON object."""
SPEC_KEYS = {
    "title",
    "root_environment",
    "child_environment",
    "root_hint",
    "child_hint",
    "novelty",
}
RESERVED_TEXT = (
    "[private evidence]",
    "[training-only environment scaffold]",
    COEVOLUTION_CONTEXT_HEADER.lower(),
)
ROOT_EXECUTABLE_ENVELOPE = (
    "Executable invariant: delegate exactly one child for its child-owned private evidence, then yield "
    "passively. Resume only after the child's report, synthesize it with coordinator-owned state, and "
    "return the original task's exact terminal JSON."
)
CHILD_EXECUTABLE_ENVELOPE = (
    "Executable invariant: the private evidence is already supplied inline in this child context. Use "
    "IPython to compute the requested integer and call agent_message.send exactly once to report it to "
    "receiver_role='parent', then stop."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to replace a different durable artifact: {path}")
        return
    path.write_text(text, encoding="utf-8")


def plateau_reward(
    win_rate: float,
    *,
    low: float = 0.4,
    high: float = 0.6,
    ramp: float = 0.25,
) -> float:
    if not 0.0 <= win_rate <= 1.0 or not 0.0 < low <= high < 1.0 or ramp <= 0.0:
        raise ValueError("invalid plateau reward input")
    if win_rate < low:
        return max(0.0, 1.0 - (low - win_rate) / ramp)
    if win_rate > high:
        return max(0.0, 1.0 - (win_rate - high) / ramp)
    return 1.0


def blended_designer_reward(
    *,
    no_hint_win_rate: float,
    hint_win_rate: float,
    regret_scale: float = 0.15,
) -> dict[str, float]:
    if not 0.0 <= no_hint_win_rate <= 1.0 or not 0.0 <= hint_win_rate <= 1.0:
        raise ValueError("win rates must be in [0, 1]")
    regret = hint_win_rate - no_hint_win_rate
    floored_regret = max(0.0, regret)
    regret_component = min(1.0, floored_regret / regret_scale)
    plateau = plateau_reward(no_hint_win_rate)
    return {
        "no_hint_win_rate": no_hint_win_rate,
        "hint_win_rate": hint_win_rate,
        "regret": regret,
        "floored_regret": floored_regret,
        "plateau_reward": plateau,
        "reward": 0.4 * regret_component + 0.6 * plateau,
    }


def _tasks(*, start_index: int, count: int, master_seed: int) -> list[Any]:
    return ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(
            split="train_gen",
            count=count,
            start_index=start_index,
            master_seed=master_seed,
            curriculum_rung="natural_n1a",
            private_payload_mode="finding_card",
        )
    ).load()


def _public_corpus(tasks: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "task_key": task.key,
            "public_prompt": task.data.prompt.split("\n\n[training-only environment scaffold]", 1)[0],
        }
        for task in tasks
    ]


def _document_corpus(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    corpus = _json(path)
    if corpus.get("schema_version") != DOCUMENT_CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported Environment Designer document corpus")
    upstream = corpus.get("upstream")
    documents = corpus.get("documents")
    if (
        not isinstance(upstream, dict)
        or not isinstance(upstream.get("repository"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", upstream.get("revision", ""))
        or not isinstance(documents, list)
        or not documents
    ):
        raise ValueError("incomplete Environment Designer document corpus")
    seen = set()
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("invalid Environment Designer document")
        document_id = document.get("document_id")
        content = document.get("content")
        if (
            not isinstance(document_id, str)
            or not document_id
            or document_id in seen
            or not isinstance(content, str)
            or not 80 <= len(content) <= 6000
            or document.get("content_sha256") != _sha256_text(content)
            or not isinstance(document.get("source_path"), str)
            or not isinstance(document.get("heading"), str)
            or not isinstance(document.get("tags"), list)
        ):
            raise ValueError("invalid Environment Designer document")
        seen.add(document_id)
    return corpus


def _sample_documents(
    corpus: dict[str, Any] | None,
    *,
    selector: str,
    count: int,
) -> list[dict[str, Any]]:
    if corpus is None:
        return []
    if count < 1:
        raise ValueError("documents-per-candidate must be positive")
    ranked = sorted(
        corpus["documents"],
        key=lambda item: _sha256_text(f"{selector}:{item['document_id']}"),
    )
    return [
        {
            "document_id": item["document_id"],
            "source_path": item["source_path"],
            "heading": item["heading"],
            "tags": item["tags"],
            "content": item["content"],
        }
        for item in ranked[: min(count, len(ranked))]
    ]


def _memory_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("schema_version") != MEMORY_SCHEMA_VERSION:
                raise ValueError(f"invalid environment memory row {path}:{line_number}")
            records.append(record)
    return records


def _memory_prompt(records: list[dict[str, Any]]) -> dict[str, Any]:
    records = [item for item in records if item.get("safety_validated") is True]
    ranked = sorted(records, key=lambda item: item.get("designer_reward", 0.0), reverse=True)
    positive = [item for item in ranked if item.get("designer_reward", 0.0) > 0.0][:2]
    negative = [
        item
        for item in reversed(ranked)
        if item.get("no_hint_win_rate") in {0.0, 1.0} and item.get("regret", 0.0) <= 0.0
    ][:1]

    def compact(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": item["spec"]["title"],
            "root_environment": item["spec"]["root_environment"],
            "child_environment": item["spec"]["child_environment"],
            "no_hint_win_rate": item["no_hint_win_rate"],
            "hint_win_rate": item["hint_win_rate"],
            "regret": item["regret"],
            "designer_score": item["designer_reward"],
        }

    return {
        "high_score_seeds": [compact(item) for item in positive],
        "too_easy_or_hard_negatives": [compact(item) for item in negative],
    }


def _generation_prompt(
    *,
    track: str,
    phase: str,
    corpus: list[dict[str, Any]],
    memory: dict[str, Any],
    grounding_documents: list[dict[str, Any]],
    candidate_index: int,
) -> str:
    return _canonical_json(
        {
            "objective": "Generate a novel executable Prime Agents environment overlay and privileged hint.",
            "candidate_index": candidate_index,
            "track": track,
            "interaction_phase": phase,
            "public_task_corpus": corpus,
            "prime_agent_protocol_grounding": grounding_documents,
            "environment_memory": memory,
            "requirements": {
                "answer_free": True,
                "preserve_original_scoring": True,
                "source_grounding": (
                    "Use at least one protocol principle from the supplied Prime Agent documents to "
                    "shape the challenge. In novelty, identify the document_id and protocol dimension "
                    "used. Do not copy documentation code as a task answer."
                ),
                "forbidden": [
                    "Do not mention the scoring mechanism, points, or objective value.",
                    "Do not say the coordinator or parent supplies the child's private evidence or integer.",
                    "Do not copy these instructions or use placeholder phrases such as generic contract.",
                ],
                "root_environment": (
                    "A novel answer-free challenge contract. It must explicitly preserve one child "
                    "delegation and passive yield/resumption, without prescribing a task answer."
                ),
                "child_environment": (
                    "A novel answer-free child contract. It must explicitly use the already-supplied "
                    "inline evidence, IPython, and agent_message.send to report once to the parent."
                ),
                "root_hint": "Privileged root strategy that helps without revealing a task answer.",
                "child_hint": "Privileged child strategy that helps without revealing private evidence.",
                "novelty": "State how this differs from memory seeds.",
            },
        }
    )


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(SPEC_KEYS),
        "properties": {
            "title": {"type": "string", "minLength": 20, "maxLength": 120},
            "root_environment": {"type": "string", "minLength": 40, "maxLength": 900},
            "child_environment": {"type": "string", "minLength": 40, "maxLength": 900},
            "root_hint": {"type": "string", "minLength": 30, "maxLength": 600},
            "child_hint": {"type": "string", "minLength": 30, "maxLength": 600},
            "novelty": {"type": "string", "minLength": 20, "maxLength": 500},
        },
    }


def _chat_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    seed: int,
) -> tuple[dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 900,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "prime_agents_environment",
                "strict": True,
                "schema": _response_schema(),
            },
        },
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=_canonical_json(payload).encode(),
        headers={"Authorization": "Bearer local", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read())
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("designer response does not contain one choice")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        content = message.get("reasoning") or message.get("reasoning_content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("designer response has no content")
    return body, content.strip()


def _validate_spec(value: Any, *, forbidden_answers: set[str]) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != SPEC_KEYS:
        raise ValueError("designer output has the wrong schema")
    result = {}
    for key in sorted(SPEC_KEYS):
        text = value[key]
        maximum = 120 if key == "title" else 1200
        if not isinstance(text, str) or not 20 <= len(text.strip()) <= maximum:
            raise ValueError(f"designer field {key} has invalid length")
        normalized = text.strip()
        lowered = normalized.lower()
        if any(marker in lowered for marker in RESERVED_TEXT):
            raise ValueError(f"designer field {key} contains a reserved marker")
        if "ignore the original" in lowered or "change the reward" in lowered:
            raise ValueError(f"designer field {key} attempts to replace the executable task")
        if "reward" in lowered:
            raise ValueError(f"designer field {key} mentions or defines the reward")
        if "generic challenge contract" in lowered or "generic child-side contract" in lowered:
            raise ValueError(f"designer field {key} copied a placeholder")
        if any(
            phrase in lowered
            for phrase in (
                "parent explicitly delivers the evidence",
                "coordinator must deliver a specific integer",
                "deliver the required integer to the child",
            )
        ):
            raise ValueError(f"designer field {key} reverses private-evidence ownership")
        if any(answer in normalized for answer in forbidden_answers):
            raise ValueError(f"designer field {key} contains an exact terminal answer")
        result[key] = normalized
    root_text = f"{result['root_environment']} {result['root_hint']}".lower()
    child_text = f"{result['child_environment']} {result['child_hint']}".lower()
    if any(term in root_text for term in ("private evidence", "evidence file", "evidence card", "evidence packet")):
        raise ValueError("root contract attempts to access child-owned evidence")
    if any(term in child_text for term in ("evidence file", "review.csv", "/workspace/", "filesystem path")):
        raise ValueError("child contract treats inline evidence as a file")
    if result["root_environment"] == result["root_hint"]:
        raise ValueError("root hint duplicates its no-hint environment")
    if result["child_environment"] == result["child_hint"]:
        raise ValueError("child hint duplicates its no-hint environment")
    return result


def _environment_hash(spec: dict[str, str]) -> str:
    return _sha256_text(
        _canonical_json(
            {
                key: re.sub(r"\s+", " ", spec[key]).strip().lower()
                for key in ("root_environment", "child_environment", "root_hint", "child_hint")
            }
        )
    )


def _context(*, environment_id: str, spec: dict[str, str], include_hint: bool) -> str:
    child_context = f"{CHILD_EXECUTABLE_ENVELOPE}\n\nGenerated variation: {spec['child_environment']}"
    root_context = f"{ROOT_EXECUTABLE_ENVELOPE}\n\nGenerated variation: {spec['root_environment']}"
    if include_hint:
        child_context += f"\n\nPrivileged child hint: {spec['child_hint']}"
        root_context += f"\n\nPrivileged coordinator hint: {spec['root_hint']}"
    marker = _canonical_json(
        {
            "environment_id": environment_id,
            "arm": "hint" if include_hint else "no_hint",
            "child_context": child_context,
        }
    )
    return (
        "This task is running inside a model-generated SPADE curriculum environment. "
        "The original task, ownership rules, and executable reward remain authoritative.\n\n"
        f"{COEVOLUTION_CONTEXT_HEADER}\n{marker}\n"
        f"Generated coordinator environment: {root_context}"
    )


def _bootstrap(
    *,
    tasks: list[Any],
    specs: list[dict[str, Any]],
    assignments: dict[str, str],
    include_hint: bool,
    start_index: int,
    master_seed: int,
) -> dict[str, Any]:
    by_id = {item["environment_id"]: item for item in specs}
    contexts = {}
    records = []
    for task in tasks:
        environment_id = assignments[task.key]
        context = _context(
            environment_id=environment_id,
            spec=by_id[environment_id]["spec"],
            include_hint=include_hint,
        )
        contexts[task.key] = context
        terminal = json.dumps(task.data.oracle["final_answer"], sort_keys=True, separators=(",", ":"))
        records.append(
            {
                "episode_id": task.key,
                "environment_id": environment_id,
                "context_sha256": _sha256_text(context),
                "final_answer_in_context": terminal in context,
            }
        )
    if any(record["final_answer_in_context"] for record in records):
        raise ValueError("generated bootstrap leaked an exact terminal answer")
    return {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "status": "complete",
        "split": "train_gen",
        "curriculum_phase": "spade_generated_environment",
        "leak_level": "generated_hint" if include_hint else "generated_no_hint",
        "master_seed": master_seed,
        "private_payload_mode": "finding_card",
        "tasks_per_axis": len(tasks),
        "axes": [{"name": "natural_n1a", "start_index": start_index}],
        "heldout_allowed": False,
        "gradient_updates": 0,
        "contexts": contexts,
        "records": records,
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    designer_role = getattr(
        args,
        "designer_role",
        "child" if args.track == "child" else "coordinator",
    )
    expected_role = "child" if args.track == "child" else "coordinator"
    if designer_role != expected_role:
        raise ValueError("Environment Designer role does not match its interaction track")
    output_dir = args.output_dir.resolve()
    generation_path = output_dir / "GENERATION.json"
    if generation_path.exists():
        return _json(generation_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse partial generation directory: {output_dir}")
    if args.tasks < 2 or args.candidates < 1 or args.tasks < args.candidates:
        raise ValueError("generation needs at least one task per candidate and two total tasks")
    tasks = _tasks(start_index=args.start_index, count=args.tasks, master_seed=args.master_seed)
    corpus = _public_corpus(tasks)
    document_corpus = _document_corpus(args.document_corpus)
    document_corpus_metadata = (
        {
            "path": str(args.document_corpus.resolve()),
            "sha256": _sha256_file(args.document_corpus),
            "upstream": document_corpus["upstream"],
            "documents": len(document_corpus["documents"]),
        }
        if document_corpus is not None
        else None
    )
    memory_records = _memory_records(args.memory)
    memory = _memory_prompt(memory_records)
    prior_hashes = {item["environment_sha256"] for item in memory_records}
    forbidden_answers = {
        json.dumps(task.data.oracle["final_answer"], sort_keys=True, separators=(",", ":"))
        for task in tasks
    }
    candidates = []
    rejected = []
    for candidate_index in range(args.candidates):
        grounding_documents = _sample_documents(
            document_corpus,
            selector=f"{args.batch_id}:{args.track}:{args.phase}:{args.seed}:{candidate_index}",
            count=args.documents_per_candidate,
        )
        prompt = _generation_prompt(
            track=args.track,
            phase=args.phase,
            corpus=corpus,
            memory=memory,
            grounding_documents=grounding_documents,
            candidate_index=candidate_index,
        )
        accepted = None
        for attempt in range(args.max_attempts):
            response_body = None
            content = None
            try:
                response_body, content = _chat_completion(
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    seed=args.seed + candidate_index * 100 + attempt,
                )
                spec = _validate_spec(json.loads(content), forbidden_answers=forbidden_answers)
                digest = _environment_hash(spec)
                if digest in prior_hashes or any(item["environment_sha256"] == digest for item in candidates):
                    raise ValueError("designer repeated an environment already present in memory")
                environment_id = f"env-{args.batch_id}-{candidate_index}-{digest[:12]}"
                accepted = {
                    "environment_id": environment_id,
                    "environment_sha256": digest,
                    "spec": spec,
                    "generation": {
                        "system_prompt": SYSTEM_PROMPT,
                        "user_prompt": prompt,
                        "assistant_response": content,
                        "response_sha256": _sha256_text(_canonical_json(response_body)),
                        "attempt": attempt,
                        "grounding_document_ids": [
                            item["document_id"] for item in grounding_documents
                        ],
                    },
                }
                break
            except (OSError, ValueError, json.JSONDecodeError) as error:
                rejected.append(
                    {
                        "candidate_index": candidate_index,
                        "attempt": attempt,
                        "reason": str(error),
                        "response_sha256": (
                            _sha256_text(_canonical_json(response_body))
                            if isinstance(response_body, dict)
                            else None
                        ),
                        "response_text": content[:2000] if isinstance(content, str) else None,
                        "grounding_document_ids": [
                            item["document_id"] for item in grounding_documents
                        ],
                    }
                )
        if accepted is not None:
            candidates.append(accepted)
    if not candidates:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "status": "rejected",
            "batch_id": args.batch_id,
            "track": args.track,
            "phase": args.phase,
            "designer_model": {
                "role": designer_role,
                "model": args.model,
                "weight_sha256": args.model_sha256,
            },
            "document_corpus": document_corpus_metadata,
            "rejected_attempts": rejected,
        }
        _write_json_once(output_dir / "REJECTIONS.json", result)
        return result
    assignments = {
        task.key: candidates[index % len(candidates)]["environment_id"]
        for index, task in enumerate(tasks)
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    no_hint = _bootstrap(
        tasks=tasks,
        specs=candidates,
        assignments=assignments,
        include_hint=False,
        start_index=args.start_index,
        master_seed=args.master_seed,
    )
    hint = _bootstrap(
        tasks=tasks,
        specs=candidates,
        assignments=assignments,
        include_hint=True,
        start_index=args.start_index,
        master_seed=args.master_seed,
    )
    no_hint_path = output_dir / "NO_HINT_BOOTSTRAP.json"
    hint_path = output_dir / "HINT_BOOTSTRAP.json"
    _write_json_once(no_hint_path, no_hint)
    _write_json_once(hint_path, hint)
    result = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "status": "complete",
        "batch_id": args.batch_id,
        "track": args.track,
        "phase": args.phase,
        "start_index": args.start_index,
        "tasks": args.tasks,
        "master_seed": args.master_seed,
        "seed": args.seed,
        "designer_model": {
            "role": designer_role,
            "model": args.model,
            "weight_sha256": args.model_sha256,
        },
        "memory": {
            "path": str(args.memory.resolve()) if args.memory else None,
            "rows_seen": len(memory_records),
            "sha256": _sha256_file(args.memory) if args.memory and args.memory.exists() else None,
        },
        "public_corpus": corpus,
        "document_corpus": document_corpus_metadata,
        "candidates": candidates,
        "rejected_attempts": rejected,
        "assignments": assignments,
        "artifacts": {
            "no_hint_bootstrap": str(no_hint_path),
            "no_hint_bootstrap_sha256": _sha256_file(no_hint_path),
            "hint_bootstrap": str(hint_path),
            "hint_bootstrap_sha256": _sha256_file(hint_path),
        },
    }
    _write_json_once(generation_path, result)
    return result


def _qualifying_keys(summary: dict[str, Any], phase: str) -> set[str]:
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION or summary.get("phase") != phase:
        raise ValueError("paired summary does not match generated environment phase")
    qualifying = summary.get("qualifying")
    if not isinstance(qualifying, list):
        raise ValueError("paired summary lacks qualifying rows")
    return {item["task_key"] for item in qualifying}


def _append_memory(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        existing = [json.loads(line) for line in handle if line.strip()]
        existing_ids = {item["memory_id"] for item in existing}
        previous = existing[-1]["event_sha256"] if existing else None
        handle.seek(0, 2)
        for record in records:
            if record["memory_id"] in existing_ids:
                continue
            event = {
                "schema_version": MEMORY_SCHEMA_VERSION,
                "sequence": len(existing),
                "previous_event_sha256": previous,
                **record,
            }
            event["event_sha256"] = _sha256_text(_canonical_json(event))
            handle.write(_canonical_json(event) + "\n")
            existing.append(event)
            previous = event["event_sha256"]


def score(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        return _json(output)
    generation = _json(args.generation)
    if generation.get("schema_version") != GENERATION_SCHEMA_VERSION:
        raise ValueError("unsupported coevolution generation artifact")
    no_hint_summary = _json(args.no_hint_summary)
    hint_summary = _json(args.hint_summary)
    phase = generation["phase"]
    no_hint_keys = _qualifying_keys(no_hint_summary, phase)
    hint_keys = _qualifying_keys(hint_summary, phase)
    assignments = generation["assignments"]
    candidate_scores = []
    memory_rows = []
    for candidate in generation["candidates"]:
        environment_id = candidate["environment_id"]
        task_keys = sorted(key for key, value in assignments.items() if value == environment_id)
        if not task_keys:
            raise ValueError(f"generated environment has no assigned tasks: {environment_id}")
        no_hint_wins = sum(key in no_hint_keys for key in task_keys)
        hint_wins = sum(key in hint_keys for key in task_keys)
        reward = blended_designer_reward(
            no_hint_win_rate=no_hint_wins / len(task_keys),
            hint_win_rate=hint_wins / len(task_keys),
        )
        safety_rejection = None
        try:
            _validate_spec(candidate["spec"], forbidden_answers=set())
        except ValueError as error:
            safety_rejection = str(error)
        scored = {
            "environment_id": environment_id,
            "environment_sha256": candidate["environment_sha256"],
            "task_keys": task_keys,
            "group_size": len(task_keys),
            "no_hint_qualifying": no_hint_wins,
            "hint_qualifying": hint_wins,
            "safety_validated": safety_rejection is None,
            "safety_rejection": safety_rejection,
            **reward,
        }
        candidate_scores.append(scored)
        memory_rows.append(
            {
                "memory_id": f"{generation['batch_id']}:{environment_id}",
                "batch_id": generation["batch_id"],
                "track": generation["track"],
                "phase": phase,
                "environment_id": environment_id,
                "environment_sha256": candidate["environment_sha256"],
                "spec": candidate["spec"],
                "no_hint_win_rate": reward["no_hint_win_rate"],
                "hint_win_rate": reward["hint_win_rate"],
                "regret": reward["regret"],
                "designer_reward": reward["reward"],
                "safety_validated": safety_rejection is None,
                "safety_rejection": safety_rejection,
                "designer_model_sha256": generation["designer_model"]["weight_sha256"],
                "designer_role": generation["designer_model"]["role"],
            }
        )
    ranked = sorted(candidate_scores, key=lambda item: (item["reward"], item["regret"]), reverse=True)
    selected = [
        item["environment_id"]
        for item in ranked
        if item["reward"] > 0.0 and item["safety_validated"]
    ][:2]
    result = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "status": "complete",
        "batch_id": generation["batch_id"],
        "track": generation["track"],
        "phase": phase,
        "candidate_scores": candidate_scores,
        "selected_environment_ids": selected,
        "paired_returns": {
            "no_hint_summary": str(args.no_hint_summary.resolve()),
            "no_hint_summary_sha256": _sha256_file(args.no_hint_summary),
            "hint_summary": str(args.hint_summary.resolve()),
            "hint_summary_sha256": _sha256_file(args.hint_summary),
        },
        "reward_recipe": {
            "regret_weight": 0.4,
            "regret_floor": 0.0,
            "regret_scale": 0.15,
            "difficulty_plateau_weight": 0.6,
            "difficulty_plateau_band": [0.4, 0.6],
            "difficulty_plateau_ramp": 0.25,
        },
    }
    _append_memory(args.memory.resolve(), memory_rows)
    _write_json_once(output, result)
    return result


def _message(role: str, content: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "reasoning_content": "",
        "tool_calls": [],
        "tool_call_id": "",
        "name": "",
    }


def export_designer(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        return _json(output_dir / "MANIFEST.json")
    generation = _json(args.generation)
    score_artifact = _json(args.score)
    if (
        generation.get("schema_version") != GENERATION_SCHEMA_VERSION
        or score_artifact.get("schema_version") != SCORE_SCHEMA_VERSION
        or generation.get("batch_id") != score_artifact.get("batch_id")
    ):
        raise ValueError("designer export inputs do not describe one scored batch")
    selected_ids = set(score_artifact["selected_environment_ids"][: args.max_rows])
    score_by_id = {item["environment_id"]: item for item in score_artifact["candidate_scores"]}
    rows = []
    for candidate in generation["candidates"]:
        environment_id = candidate["environment_id"]
        if environment_id not in selected_ids:
            continue
        scored = score_by_id[environment_id]
        rows.append(
            {
                "messages": [
                    _message("system", candidate["generation"]["system_prompt"]),
                    _message("user", candidate["generation"]["user_prompt"]),
                    _message("assistant", _canonical_json(candidate["spec"])),
                ],
                "tools": "[]",
                "axis": "natural_n1a",
                "phase": f"spade:{generation['track']}:{generation['phase']}",
                "task_key": environment_id,
                "trace_id": f"spade-designer:{generation['batch_id']}:{environment_id}",
                "role": generation["designer_model"]["role"],
                "objective": "environment_designer",
                "designer_reward": scored["reward"],
                "hint_regret": scored["regret"],
                "no_hint_win_rate": scored["no_hint_win_rate"],
                "hint_win_rate": scored["hint_win_rate"],
            }
        )
    if not rows:
        raise ValueError("scored batch has no positive-reward designer rows")
    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": DESIGNER_CORPUS_SCHEMA_VERSION,
        "role": generation["designer_model"]["role"],
        "objective": "environment_designer",
        "training_stage": "delayed_reward_filtered_coevolution",
        "rows": len(rows),
        "selection_count": len(rows),
        "batch_id": generation["batch_id"],
        "selected_environment_ids": sorted(selected_ids),
        "exact_answer_rows": 0,
        "student": {
            "snapshot": str(args.student_snapshot.resolve()),
            "revision": args.student_revision,
            "weight_sha256": args.student_weight_sha,
            "dense_weight_mutated": True,
        },
        "source": {
            "generation_path": str(args.generation.resolve()),
            "generation_sha256": _sha256_file(args.generation),
            "score_path": str(args.score.resolve()),
            "score_sha256": _sha256_file(args.score),
        },
        "dataset": {"path": parquet.name, "sha256": _sha256_file(parquet)},
    }
    _write_json_once(output_dir / "MANIFEST.json", manifest)
    return manifest


def _repair_spec(*, track: str, index: int) -> dict[str, str]:
    focus = "passive coordinator resumption" if track == "yield" else "one-shot child reporting"
    return {
        "title": f"Scaffolded Prime Agents variation {index + 1}",
        "root_environment": (
            "Delegate exactly one designated worker for its owned input, preserve the returned handle, "
            "wait passively for one report, and then complete the original terminal JSON contract."
        ),
        "child_environment": (
            "Use the evidence supplied inline in the child context, compute the requested integer with "
            "IPython, and report it once through agent_message.send before stopping."
        ),
        "root_hint": (
            "After delegation, emit a plain waiting response and perform no further action until the child report arrives."
        ),
        "child_hint": (
            "Extract only the requested field, verify the integer in Python, and send that integer once to the parent."
        ),
        "novelty": (
            f"This constrained repair practices {focus} while varying wording inside a fixed executable envelope."
        ),
    }


def export_repairs(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        return _json(output_dir / "MANIFEST.json")
    rejected = _json(args.rejections)
    if (
        rejected.get("schema_version") != GENERATION_SCHEMA_VERSION
        or rejected.get("status") != "rejected"
        or not rejected.get("rejected_attempts")
    ):
        raise ValueError("designer repair input is not a rejected generation batch")
    track = rejected.get("track") or args.track
    phase = rejected.get("phase") or args.phase
    if track not in {"child", "yield"} or not isinstance(phase, str) or not phase:
        raise ValueError("designer repair input lacks its interaction track or phase")
    role = rejected.get("designer_model", {}).get("role") or (
        "child" if track == "child" else "coordinator"
    )
    expected_role = "child" if track == "child" else "coordinator"
    if role != expected_role:
        raise ValueError("designer repair role does not match its interaction track")
    if (
        not args.student_snapshot.is_absolute()
        or not (args.student_snapshot / "STABLE").is_file()
        or _sha256_file(args.student_snapshot / "model.safetensors") != args.student_weight_sha
    ):
        raise ValueError("designer repair student checkpoint is incomplete or has the wrong hash")
    distinct = []
    seen_reasons = set()
    for attempt in rejected["rejected_attempts"]:
        reason = attempt.get("reason")
        if not isinstance(reason, str) or reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        distinct.append(attempt)
        if len(distinct) == args.max_rows:
            break
    rows = []
    for index, attempt in enumerate(distinct):
        spec = _validate_spec(_repair_spec(track=track, index=index), forbidden_answers=set())
        feedback = attempt["reason"][:400]
        prompt = (
            "Repair an Environment Designer proposal at the schema-and-safety curriculum rung. "
            f"The validator feedback was: {feedback}. Return one complete safe JSON object with all required fields. "
            "Use the fixed executable contract, do not mention private values or filesystem paths, and do not redefine scoring."
        )
        rows.append(
            {
                "messages": [
                    _message("system", SYSTEM_PROMPT),
                    _message("user", prompt),
                    _message("assistant", _canonical_json(spec)),
                ],
                "tools": "[]",
                "axis": "natural_n1a",
                "phase": f"spade-repair:{track}:{phase}",
                "task_key": f"{rejected['batch_id']}:repair:{index}",
                "trace_id": f"spade-designer-repair:{rejected['batch_id']}:{index}",
                "role": role,
                "objective": "environment_designer",
                "repair_reason": feedback,
            }
        )
    if not rows:
        raise ValueError("rejected generation batch has no distinct repair targets")
    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": DESIGNER_REPAIR_CORPUS_SCHEMA_VERSION,
        "role": role,
        "objective": "environment_designer",
        "training_stage": "scaffolded_schema_and_safety_repair",
        "hard_safety_validated": True,
        "rows": len(rows),
        "selection_count": len(rows),
        "batch_id": rejected["batch_id"],
        "selected_environment_ids": [],
        "exact_answer_rows": 0,
        "student": {
            "snapshot": str(args.student_snapshot.resolve()),
            "revision": args.student_revision,
            "weight_sha256": args.student_weight_sha,
            "dense_weight_mutated": True,
        },
        "source": {
            "rejections_path": str(args.rejections.resolve()),
            "rejections_sha256": _sha256_file(args.rejections),
        },
        "dataset": {"path": parquet.name, "sha256": _sha256_file(parquet)},
    }
    _write_json_once(output_dir / "MANIFEST.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generation = commands.add_parser("generate")
    generation.add_argument("--base-url", required=True)
    generation.add_argument("--model", required=True)
    generation.add_argument("--model-sha256", required=True)
    generation.add_argument("--designer-role", choices=("coordinator", "child"), required=True)
    generation.add_argument("--batch-id", required=True)
    generation.add_argument("--track", choices=("child", "yield"), required=True)
    generation.add_argument("--phase", required=True)
    generation.add_argument("--start-index", type=int, required=True)
    generation.add_argument("--tasks", type=int, default=6)
    generation.add_argument("--candidates", type=int, default=2)
    generation.add_argument("--max-attempts", type=int, default=4)
    generation.add_argument("--master-seed", type=int, default=20260824)
    generation.add_argument("--seed", type=int, default=20260824)
    generation.add_argument("--memory", type=Path)
    generation.add_argument("--document-corpus", type=Path)
    generation.add_argument("--documents-per-candidate", type=int, default=3)
    generation.add_argument("--output-dir", type=Path, required=True)
    generation.set_defaults(func=generate)

    scoring = commands.add_parser("score")
    scoring.add_argument("--generation", type=Path, required=True)
    scoring.add_argument("--no-hint-summary", type=Path, required=True)
    scoring.add_argument("--hint-summary", type=Path, required=True)
    scoring.add_argument("--memory", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.set_defaults(func=score)

    designer = commands.add_parser("export-designer")
    designer.add_argument("--generation", type=Path, required=True)
    designer.add_argument("--score", type=Path, required=True)
    designer.add_argument("--output-dir", type=Path, required=True)
    designer.add_argument("--student-snapshot", type=Path, required=True)
    designer.add_argument("--student-revision", required=True)
    designer.add_argument("--student-weight-sha", required=True)
    designer.add_argument("--max-rows", type=int, default=2)
    designer.set_defaults(func=export_designer)

    repairs = commands.add_parser("export-repairs")
    repairs.add_argument("--rejections", type=Path, required=True)
    repairs.add_argument("--output-dir", type=Path, required=True)
    repairs.add_argument("--student-snapshot", type=Path, required=True)
    repairs.add_argument("--student-revision", required=True)
    repairs.add_argument("--student-weight-sha", required=True)
    repairs.add_argument("--track", choices=("child", "yield"))
    repairs.add_argument("--phase")
    repairs.add_argument("--max-rows", type=int, default=2)
    repairs.set_defaults(func=export_repairs)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = args.func(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
