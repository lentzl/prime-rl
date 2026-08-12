#!/usr/bin/env python3
"""Generate a semantically varied paired ownership corpus for Prime Agent.

The batch is intentionally about one invariant only:

    ownership follows the decomposition contract, not the presence of a path.

Each of 32 semantic tasks is emitted twice:
- child-owned: retain coordinator-local state, spawn exactly one named child,
  keep the delegated resource path only inside the rlm(...) prompt, retain the
  admission handle, then yield without polling;
- coordinator-owned: keep the same resource kind/operation local, do not call
  rlm, inspect the resource directly, and return the result.

The 64 resulting examples vary resource type, operation, child role, local
state, and natural-language ownership phrasing. They contain no fabricated
reasoning_content; this is deliberate because thinking-mode teacher traces
should use authentic model-generated reasoning.

This is a deterministic synthetic SFT-format corpus. The synthetic resource
contents are stored in metadata so the examples can later be converted into
an executable environment if desired.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

from ipython_sft_export_utils import IPYTHON_TOOL, tool_call
from subagent_communication_v1.taskset import SYSTEM_PROMPT


DATASET = "ownership-invariant-pairs-v1"
DEFAULT_SEED = 20260812

DELEGATE_TEMPLATES = (
    "Keep {local_var}={local_val} as coordinator-owned state. Delegate the work on {path} "
    "to a child named {child}: {operation}. The child owns that resource access; do not "
    "inspect or parse {path} in the coordinator. Retain the admission handle and yield after the spawn.",
    "Split this task by ownership. You retain {local_var}={local_val}. Have {child} take "
    "responsibility for {path} and {operation}. Do not open the child-owned resource yourself. "
    "Keep the child handle and wait for its explicit message rather than polling.",
    "The coordinator is responsible for {local_var}={local_val}; the resource task belongs "
    "to {child}. Spawn {child} to {operation} at {path}. The path may appear inside the child "
    "prompt, but the coordinator must not read it. Preserve the handle and end the turn after successful admission.",
    "Offload only the resource shard. Have {child} {operation} at {path}. Keep "
    "{local_var}={local_val} locally. Do not duplicate the delegated work in the parent. "
    "Bind the returned RLM handle and then yield for the child's explicit reply.",
)

DIRECT_TEMPLATES = (
    "Do not delegate this task. Keep {local_var}={local_val}, read {path} yourself in "
    "persistent IPython, and {operation}. Return JSON with keys local_state and result.",
    "This resource is coordinator-owned. Retain {local_var}={local_val} and inspect {path} "
    "directly to {operation}. No child is needed. Return only the requested JSON object.",
    "Keep the whole task local: preserve {local_var}={local_val}, access {path} in the "
    "coordinator, and {operation}. Do not call rlm. Return JSON with local_state and result.",
    "No delegation is assigned here. In the coordinator kernel, retain {local_var}={local_val}, "
    "open {path}, and {operation}. Return only the requested JSON object with local_state and result.",
)

RESOURCE_KINDS = (
    ("json_sum", "ledger-worker"),
    ("csv_amount_total", "table-worker"),
    ("text_keyword_count", "text-worker"),
    ("markdown_heading_count", "outline-worker"),
    ("log_error_count", "log-worker"),
    ("python_def_count", "code-worker"),
    ("json_max_value", "metrics-worker"),
    ("sha256_prefix", "hash-worker"),
)

LOCAL_VARS = ("priority", "multiplier", "batch_id", "threshold")


def _tool_messages(call_id: str, code: str, output: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call(call_id, code)],
        },
        {"role": "tool", "content": output, "tool_call_id": call_id},
    ]


def _path_used_outside_rlm(code: str, path: str) -> bool:
    tree = ast.parse(code)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if path not in node.value:
            continue
        ancestor: ast.AST = node
        while ancestor in parents:
            ancestor = parents[ancestor]
            if (
                isinstance(ancestor, ast.Call)
                and isinstance(ancestor.func, ast.Name)
                and ancestor.func.id == "rlm"
            ):
                break
        else:
            return True
    return False


def _specs(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    specs: list[dict[str, Any]] = []

    for kind_index, (kind, child) in enumerate(RESOURCE_KINDS):
        for variant in range(4):
            pair_id = f"{kind}-{variant + 1:02d}"
            local_var = LOCAL_VARS[variant]
            local_val = (
                7 + kind_index,
                3 + kind_index,
                410 + kind_index * 10 + variant,
                11 + kind_index,
            )[variant]
            path = f"/workspace/synthetic/{kind}/{pair_id}"

            if kind == "json_sum":
                path += ".json"
                values = [rng.randint(-9, 30) for _ in range(6 + variant)]
                content = json.dumps(values)
                result: int | str = sum(values)
                operation = "load the JSON integer list and return its sum"
                direct_code = (
                    f"{local_var} = {local_val}\n"
                    "import json\nfrom pathlib import Path\n"
                    f"values = json.loads(Path({path!r}).read_text())\n"
                    "result = sum(values)\n"
                    f"answer = {{'local_state': {local_var}, 'result': result}}\nanswer"
                )
                child_prompt = (
                    f"Read {path} as a JSON list of integers, compute sum(values), and send only "
                    "the integer result to your parent with "
                    "await agent_message.send(str(result), receiver_role='parent'). "
                    "After the delivery receipt succeeds, stop."
                )

            elif kind == "csv_amount_total":
                path += ".csv"
                rows = [
                    {"id": f"r{i}", "amount": str(rng.randint(2, 90))}
                    for i in range(4 + variant)
                ]
                buffer = io.StringIO()
                writer = csv.DictWriter(buffer, fieldnames=["id", "amount"])
                writer.writeheader()
                writer.writerows(rows)
                content = buffer.getvalue()
                result = sum(int(row["amount"]) for row in rows)
                operation = "sum the integer values in the CSV amount column"
                direct_code = (
                    f"{local_var} = {local_val}\n"
                    "import csv\nfrom pathlib import Path\n"
                    f"rows = list(csv.DictReader(Path({path!r}).read_text().splitlines()))\n"
                    "result = sum(int(row['amount']) for row in rows)\n"
                    f"answer = {{'local_state': {local_var}, 'result': result}}\nanswer"
                )
                child_prompt = (
                    f"Read {path} as CSV, sum the integer amount column, and send only the "
                    "integer result to your parent with "
                    "await agent_message.send(str(result), receiver_role='parent'). "
                    "After delivery, stop."
                )

            elif kind == "text_keyword_count":
                path += ".txt"
                keyword = ("retry", "timeout", "cache", "stale")[variant]
                tokens = ["ok", keyword, "ready", keyword, "done"]
                if variant % 2 == 0:
                    tokens.append(keyword)
                tokens.append("stable")
                content = " ".join(tokens)
                result = content.split().count(keyword)
                operation = f"count exact whitespace-delimited occurrences of {keyword!r}"
                direct_code = (
                    f"{local_var} = {local_val}\n"
                    "from pathlib import Path\n"
                    f"text = Path({path!r}).read_text()\n"
                    f"result = text.split().count({keyword!r})\n"
                    f"answer = {{'local_state': {local_var}, 'result': result}}\nanswer"
                )
                child_prompt = (
                    f"Read {path}, count exact whitespace-delimited occurrences of {keyword!r}, "
                    "and send only the integer count to your parent with "
                    "await agent_message.send(str(result), receiver_role='parent'). "
                    "After delivery, stop."
                )

            elif kind == "markdown_heading_count":
                path += ".md"
                headings = [f"## Section {i}" for i in range(2 + variant)]
                content = "# Report\n\n" + "\ntext\n".join(headings) + "\n"
                result = sum(
                    1 for line in content.splitlines() if line.startswith("## ")
                )
                operation = "count level-2 Markdown headings"
                direct_code = (
                    f"{local_var} = {local_val}\n"
                    "from pathlib import Path\n"
                    f"lines = Path({path!r}).read_text().splitlines()\n"
                    "result = sum(1 for line in lines if line.startswith('## '))\n"
                    f"answer = {{'local_state': {local_var}, 'result': result}}\nanswer"
                )
                child_prompt = (
                    f"Read {path}, count lines beginning exactly with '## ', and send only "
                    "the integer count to your parent with "
                    "await agent_message.send(str(result), receiver_role='parent'). "
                    "After delivery, stop."
                )

            elif kind == "log_error_count":
                path += ".log"
                count = 1 + variant
                content = "\n".join(
                    ["INFO boot", "WARN slow"] + ["ERROR failed"] * count + ["INFO done"]
                )
                result = count
                operation = "count log lines whose level is ERROR"
                direct_code = (
                    f"{local_var} = {local_val}\n"
                    "from pathlib import Path\n"
                    f"lines = Path({path!r}).read_text().splitlines()\n"
                    "result = sum(1 for line in lines if line.startswith('ERROR '))\n"
                    f"answer = {{'local_state': {local_var}, 'result': result}}\nanswer"
                )
                child_prompt = (
                    f"Read {path}, count lines starting with 'ERROR ', and send only the "
                    "integer count to your parent with "
                    "await agent_message.send(str(result), receiver_role='parent'). "
                    "After delivery, stop."
                )

            elif kind == "python_def_count":
                path += ".py"
                definitions = [
                    f"def f{i}():\n    return {i}\n" for i in range(1 + variant)
                ]
                if variant % 2:
                    definitions.append("async def g():\n    return 1\n")
                content = "\n".join(definitions)
                tree = ast.parse(content)
                result = sum(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    for node in ast.walk(tree)
                )
                operation = "count function and async-function definitions in the Python source"
                direct_code = (
                    f"{local_var} = {local_val}\n"
                    "import ast\nfrom pathlib import Path\n"
                    f"tree = ast.parse(Path({path!r}).read_text())\n"
                    "result = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) "
                    "for node in ast.walk(tree))\n"
                    f"answer = {{'local_state': {local_var}, 'result': result}}\nanswer"
                )
                child_prompt = (
                    f"Read {path}, parse it with ast, count FunctionDef plus AsyncFunctionDef "
                    "nodes, and send only the integer count to your parent with "
                    "await agent_message.send(str(result), receiver_role='parent'). "
                    "After delivery, stop."
                )

            elif kind == "json_max_value":
                path += ".json"
                obj = {f"k{i}": rng.randint(-20, 70) for i in range(5 + variant)}
                content = json.dumps(obj)
                result = max(obj.values())
                operation = "find the maximum numeric value in the JSON object"
                direct_code = (
                    f"{local_var} = {local_val}\n"
                    "import json\nfrom pathlib import Path\n"
                    f"obj = json.loads(Path({path!r}).read_text())\n"
                    "result = max(obj.values())\n"
                    f"answer = {{'local_state': {local_var}, 'result': result}}\nanswer"
                )
                child_prompt = (
                    f"Read {path} as a JSON object, find max(obj.values()), and send only "
                    "that integer to your parent with "
                    "await agent_message.send(str(result), receiver_role='parent'). "
                    "After delivery, stop."
                )

            else:
                path += ".bin"
                payload = (
                    f"synthetic payload {pair_id} " + "x" * (10 + variant)
                ).encode()
                content = payload.decode()
                result = hashlib.sha256(payload).hexdigest()[:8]
                operation = (
                    "compute the first eight hexadecimal characters of the SHA-256 digest"
                )
                direct_code = (
                    f"{local_var} = {local_val}\n"
                    "import hashlib\nfrom pathlib import Path\n"
                    f"payload = Path({path!r}).read_bytes()\n"
                    "result = hashlib.sha256(payload).hexdigest()[:8]\n"
                    f"answer = {{'local_state': {local_var}, 'result': result}}\nanswer"
                )
                child_prompt = (
                    f"Read the bytes at {path}, compute "
                    "hashlib.sha256(payload).hexdigest()[:8], and send only that "
                    "eight-character string to your parent with "
                    "await agent_message.send(result, receiver_role='parent'). "
                    "After delivery, stop."
                )

            specs.append(
                {
                    "pair_id": pair_id,
                    "kind": kind,
                    "local_var": local_var,
                    "local_val": local_val,
                    "path": path,
                    "child": child,
                    "operation": operation,
                    "direct_code": direct_code,
                    "child_prompt": child_prompt,
                    "result": result,
                    "content": content,
                    "variant": variant,
                }
            )

    return specs


def build_examples(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for spec in _specs(seed):
        prompt_args = dict(spec)

        delegated_user = DELEGATE_TEMPLATES[spec["variant"]].format(**prompt_args)
        delegated_code = (
            f"{spec['local_var']} = {spec['local_val']}\n"
            f"child = await rlm({spec['child_prompt']!r}, name={spec['child']!r})"
        )
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": delegated_user},
                    *_tool_messages(
                        f"{spec['pair_id']}-delegate",
                        delegated_code,
                        "",
                    ),
                    {
                        "role": "assistant",
                        "content": f"Waiting for {spec['child']}'s explicit reply.",
                    },
                ],
                "tools": [IPYTHON_TOOL],
                "metadata": {
                    "dataset": DATASET,
                    "pair_id": spec["pair_id"],
                    "ownership": "child",
                    "resource_kind": spec["kind"],
                    "wording_variant": spec["variant"],
                    "training_scope": "first_coordinator_transition",
                    "target_invariant": "ownership_follows_decomposition_contract",
                    "no_fabricated_reasoning": True,
                    "synthetic_resource_path": spec["path"],
                    "synthetic_resource_content": spec["content"],
                    "expected_remote_result": spec["result"],
                    "role": "parent",
                },
            }
        )

        direct_user = DIRECT_TEMPLATES[spec["variant"]].format(**prompt_args)
        direct_answer = {
            "local_state": spec["local_val"],
            "result": spec["result"],
        }
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": direct_user},
                    *_tool_messages(
                        f"{spec['pair_id']}-direct",
                        spec["direct_code"],
                        repr(direct_answer),
                    ),
                    {
                        "role": "assistant",
                        "content": json.dumps(direct_answer, separators=(",", ":")),
                    },
                ],
                "tools": [IPYTHON_TOOL],
                "metadata": {
                    "dataset": DATASET,
                    "pair_id": spec["pair_id"],
                    "ownership": "coordinator",
                    "resource_kind": spec["kind"],
                    "wording_variant": spec["variant"],
                    "training_scope": "first_coordinator_transition",
                    "target_invariant": "ownership_follows_decomposition_contract",
                    "no_fabricated_reasoning": True,
                    "synthetic_resource_path": spec["path"],
                    "synthetic_resource_content": spec["content"],
                    "expected_local_result": spec["result"],
                    "role": "parent",
                },
            }
        )

    _validate(examples)
    return examples


def _validate(examples: list[dict[str, Any]]) -> None:
    if len(examples) != 64:
        raise ValueError(f"expected 64 examples, got {len(examples)}")

    pair_arms: dict[str, set[str]] = {}
    kinds: set[str] = set()

    for example in examples:
        metadata = example["metadata"]
        pair_arms.setdefault(metadata["pair_id"], set()).add(metadata["ownership"])
        kinds.add(metadata["resource_kind"])

        tool_message = next(
            message
            for message in example["messages"]
            if message.get("role") == "assistant" and message.get("tool_calls")
        )
        code = json.loads(
            tool_message["tool_calls"][0]["function"]["arguments"]
        )["code"]
        path = metadata["synthetic_resource_path"]

        if metadata["ownership"] == "child":
            if "child = await rlm(" not in code:
                raise ValueError(f"delegated example lost handle: {metadata['pair_id']}")
            if _path_used_outside_rlm(code, path):
                raise ValueError(
                    f"delegated path escapes rlm prompt: {metadata['pair_id']}"
                )
            if "Path(" in code or ".read_" in code:
                raise ValueError(
                    f"delegated example directly reads resource: {metadata['pair_id']}"
                )
        else:
            if "rlm(" in code:
                raise ValueError(
                    f"coordinator-owned control delegates: {metadata['pair_id']}"
                )
            if path not in code or "Path(" not in code:
                raise ValueError(
                    f"coordinator-owned control does not read resource: {metadata['pair_id']}"
                )

        if any("reasoning_content" in message for message in example["messages"]):
            raise ValueError(
                f"synthetic reasoning is forbidden in {metadata['pair_id']}"
            )

    if len(pair_arms) != 32 or any(
        arms != {"child", "coordinator"} for arms in pair_arms.values()
    ):
        raise ValueError("every semantic task must have one child-owned and one local arm")

    if kinds != {kind for kind, _ in RESOURCE_KINDS}:
        raise ValueError(f"resource-kind coverage mismatch: {sorted(kinds)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    examples = build_examples(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n"
            for example in examples
        )
    )

    delegated = sum(
        example["metadata"]["ownership"] == "child" for example in examples
    )
    direct = len(examples) - delegated
    kinds = sorted(
        {example["metadata"]["resource_kind"] for example in examples}
    )
    print(
        json.dumps(
            {
                "dataset": DATASET,
                "examples": len(examples),
                "pairs": len(examples) // 2,
                "delegated": delegated,
                "coordinator_owned": direct,
                "resource_kinds": kinds,
                "reasoning_content": 0,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
