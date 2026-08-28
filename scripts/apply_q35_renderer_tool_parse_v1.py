#!/usr/bin/env python3
"""Idempotently align the raw Qwen3.5 renderer parser with vLLM."""

import argparse
from pathlib import Path

OLD = '''            if end == -1:
                raw = _decode(tokenizer, ids[i + 1 :])
                tool_calls.append(
                    ParsedToolCall(
                        raw=raw,
                        token_span=(section_offset + i, section_offset + len(ids)),
                        status=ToolCallParseStatus.UNCLOSED_BLOCK,
                    )
                )
                break
            block_text = _decode(tokenizer, ids[i + 1 : end])
            span = (section_offset + i, section_offset + end + 1)
'''

NEW = '''            if end == -1:
                raw = _decode(tokenizer, ids[i + 1 :])
                # vLLM's Qwen3 parser closes a tool event as soon as it sees
                # ``</function>``; it does not require the redundant outer
                # ``</tool_call>`` marker. Match that behavior for raw-token
                # rollouts. Qwen3.5 sometimes emits a complete function block
                # and then ``<|im_end|>`` directly, and treating that as an
                # unclosed attempt makes an otherwise exact on-policy action
                # disappear before the agent can execute it.
                if not re.search(r"</function>\\s*$", raw):
                    tool_calls.append(
                        ParsedToolCall(
                            raw=raw,
                            token_span=(
                                section_offset + i,
                                section_offset + len(ids),
                            ),
                            status=ToolCallParseStatus.UNCLOSED_BLOCK,
                        )
                    )
                    break
                block_text = raw
                span = (section_offset + i, section_offset + len(ids))
                end = len(ids) - 1
            else:
                block_text = _decode(tokenizer, ids[i + 1 : end])
                span = (section_offset + i, section_offset + end + 1)
'''


def apply_parser_patch(path: Path) -> str:
    source = path.read_text()
    if source.count(NEW) == 1 and OLD not in source:
        return "already_applied"
    if source.count(OLD) != 1 or NEW in source:
        raise RuntimeError("renderer source is incompatible with the Qwen3.5 parser patch")
    path.write_text(source.replace(OLD, NEW, 1))
    return "applied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(apply_parser_patch(args.path))


if __name__ == "__main__":
    main()
