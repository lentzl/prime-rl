#!/usr/bin/env python3
"""Verify the Qwen3.5 raw-token tool parser accepts a closed function body."""

from renderers.base import ToolCallParseStatus
from renderers.parsing import _parse_xml_tool_calls


class _Tokenizer:
    def __init__(self, pieces: dict[int, str]):
        self.pieces = pieces

    def decode(self, ids, *, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(self.pieces[token_id] for token_id in ids)


def _parse(body: str):
    tokenizer = _Tokenizer({1: "<tool_call>", 2: "</tool_call>", 3: body})
    return _parse_xml_tool_calls(
        tokenizer,
        [1, 3],
        1,
        2,
        section_offset=0,
        param_index={"ipython": {"code": {"type": "string"}}},
    )


def main() -> None:
    complete = _parse(
        "\n<function=ipython><parameter=code>\n"
        "reviewer = await rlm('task', name='worker')\n"
        "</parameter></function>"
    )
    if len(complete) != 1 or complete[0].status is not ToolCallParseStatus.OK:
        raise SystemExit("complete Qwen3.5 function body was not accepted")
    if complete[0].name != "ipython" or complete[0].arguments != {
        "code": "reviewer = await rlm('task', name='worker')"
    }:
        raise SystemExit("complete Qwen3.5 function body parsed incorrectly")

    incomplete = _parse(
        "\n<function=ipython><parameter=code>\n"
        "reviewer = await rlm('task', name='worker')"
    )
    if (
        len(incomplete) != 1
        or incomplete[0].status is not ToolCallParseStatus.UNCLOSED_BLOCK
    ):
        raise SystemExit("incomplete Qwen3.5 function body was accepted")

    print("Qwen3.5 raw-token tool parser tolerance verified")


if __name__ == "__main__":
    main()
