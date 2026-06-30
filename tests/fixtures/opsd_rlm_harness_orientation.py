"""Small RLM-shaped harness-orientation environment for OPSD runs.

The tasks are intentionally simple and text-only. They exercise the episode
shape we care about: helper/library grounding, execution-style feedback,
repair, and final answer submission through a tiny multi-turn verifier loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset

import verifiers as vf
from verifiers.types import Messages, State


@dataclass(frozen=True)
class HarnessTask:
    name: str
    helper: str
    request: str
    expected: str
    demonstration: str
    feedback: str


TASKS = [
    HarnessTask(
        name="normalize_tokens_alpha",
        helper="normalize_tokens(text): lowercase text, keep alphanumeric characters, split on whitespace",
        request='Normalize this noisy record: "Alpha!! beta, ALPHA... gamma?"',
        expected="alpha beta alpha gamma",
        demonstration=(
            "Use the normalize_tokens helper for the text rule, inspect the noisy record, "
            "then submit <answer>alpha beta alpha gamma</answer>."
        ),
        feedback="The helper should own lowercase-and-strip-punctuation behavior. Submit only the normalized tokens.",
    ),
    HarnessTask(
        name="sum_even_values",
        helper="sum_even(values): add only integers divisible by 2",
        request="Use the helper to sum the even values in [3, 8, 5, 12, 7, 4].",
        expected="24",
        demonstration=(
            "Load or reconstruct sum_even, select 8, 12, and 4, compute 24, "
            "then submit <answer>24</answer>."
        ),
        feedback="Ignore odd values. The accepted answer is the numeric sum wrapped in answer tags.",
    ),
    HarnessTask(
        name="extract_doc_ids",
        helper="extract_doc_ids(text): return document ids in first-appearance order",
        request='Extract document ids from: "see doc-7, then doc-2; doc-7 is repeated".',
        expected="doc-7 doc-2",
        demonstration=(
            "Use extract_doc_ids, preserve first-appearance order, drop repeats, "
            "then submit <answer>doc-7 doc-2</answer>."
        ),
        feedback="Keep ids in first-seen order and remove duplicates before the final answer.",
    ),
    HarnessTask(
        name="join_sorted_keys",
        helper="join_sorted_keys(mapping): sort keys alphabetically and join them with spaces",
        request='Apply the helper to {"zeta": 1, "alpha": 2, "mu": 3}.',
        expected="alpha mu zeta",
        demonstration=(
            "Inspect the mapping keys, sort alpha, mu, zeta, "
            "then submit <answer>alpha mu zeta</answer>."
        ),
        feedback="Only the keys matter. Sort them alphabetically and submit the space-joined keys.",
    ),
    HarnessTask(
        name="repair_missing_suffix",
        helper="ensure_suffix(name, suffix): append suffix only when it is missing",
        request='Repair the filename "report" so it has suffix ".json".',
        expected="report.json",
        demonstration=(
            "Use ensure_suffix, observe that report lacks .json, append it once, "
            "then submit <answer>report.json</answer>."
        ),
        feedback="Do not duplicate the suffix. The final answer should be the repaired filename.",
    ),
    HarnessTask(
        name="count_marker_hits",
        helper="count_marker(lines, marker): count lines containing marker exactly",
        request='Count lines containing "TODO" in ["TODO fix", "done", "TODO test", "todo lower"].',
        expected="2",
        demonstration=(
            "Use count_marker with marker TODO, match the two uppercase TODO lines, "
            "then submit <answer>2</answer>."
        ),
        feedback="Matching is case-sensitive here. Count only the uppercase TODO occurrences.",
    ),
    HarnessTask(
        name="compose_slug",
        helper="slugify_title(title): lowercase, keep alphanumeric words, join with hyphens",
        request='Create a slug for title "RLM Harness: Tiny Steps!".',
        expected="rlm-harness-tiny-steps",
        demonstration=(
            "Use slugify_title, normalize words to rlm harness tiny steps, "
            "then submit <answer>rlm-harness-tiny-steps</answer>."
        ),
        feedback="The helper output uses hyphens, not spaces. Submit the slug in answer tags.",
    ),
    HarnessTask(
        name="state_lookup",
        helper="lookup_state(key): retrieve the current episode value for a key",
        request='The episode state has color="blue" and shape="triangle". Return shape.',
        expected="triangle",
        demonstration=(
            "Use lookup_state for key shape, retrieve triangle from the episode state, "
            "then submit <answer>triangle</answer>."
        ),
        feedback="The requested key is shape, not color. Submit the retrieved value.",
    ),
    HarnessTask(
        name="dedupe_preserve_order",
        helper="dedupe_order(items): remove repeats while preserving first occurrence",
        request='Dedupe ["red", "blue", "red", "green", "blue"].',
        expected="red blue green",
        demonstration=(
            "Use dedupe_order, keep red then blue then green, "
            "then submit <answer>red blue green</answer>."
        ),
        feedback="Preserve first occurrence order. Do not sort the values.",
    ),
    HarnessTask(
        name="parse_last_status",
        helper="last_status(events): return the status value from the final event",
        request='Events are [{"status": "queued"}, {"status": "running"}, {"status": "done"}].',
        expected="done",
        demonstration=(
            "Use last_status, inspect the final event, read status done, "
            "then submit <answer>done</answer>."
        ),
        feedback="Read the final event only. The final status is the accepted answer.",
    ),
    HarnessTask(
        name="tool_error_recovery",
        helper="safe_divide(a, b): if b is zero, return undefined instead of crashing",
        request="Recover from a divide-by-zero attempt for 9 / 0.",
        expected="undefined",
        demonstration=(
            "Use safe_divide, notice b is zero, recover by returning undefined, "
            "then submit <answer>undefined</answer>."
        ),
        feedback="This is a recovery task. Do not crash or invent a number; submit undefined.",
    ),
    HarnessTask(
        name="select_longest_word",
        helper="longest_word(words): return the longest word, using first one on ties",
        request='Choose from ["arc", "harness", "library", "run"].',
        expected="harness",
        demonstration=(
            "Use longest_word, compare lengths, harness and library tie at seven, "
            "keep the first tie and submit <answer>harness</answer>."
        ),
        feedback="On ties, keep the first longest word. Submit only that word.",
    ),
]


TASK_TEMPLATE = """You are in a tiny REPL-style harness.

Persistent library card:
- {helper}

Task:
{request}

Use the helper behavior, recover if feedback corrects you, and submit exactly one final answer:
<answer>...</answer>
"""

RECOVERY_PROMPT = "Repair the attempt using the feedback and submit the final answer now."


class OPSDRLMHarnessOrientationEnv(vf.MultiTurnEnv):
    @vf.stop
    async def final_feedback_seen(self, state: State) -> bool:
        return len(state["trajectory"]) >= 3

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        turn = len(state["trajectory"])
        if turn == 1:
            return [vf.UserMessage(content=f"Execution feedback:\n- {state['info']['feedback']}")]
        return [vf.UserMessage(content=RECOVERY_PROMPT)]


def load_environment() -> vf.Environment:
    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": TASK_TEMPLATE.format(helper=task.helper, request=task.request),
                    }
                ],
                "answer": task.demonstration,
                "info": {
                    "name": task.name,
                    "expected": task.expected,
                    "feedback": task.feedback,
                },
            }
            for task in TASKS
        ]
    )
    parser = vf.XMLParser(["answer"], answer_field="answer")

    def final_answer(parser, completion, info, **kwargs) -> float:
        parsed = parser.parse_answer(completion) or ""
        expected = info["expected"]
        return 1.0 if " ".join(parsed.lower().split()) == expected else 0.0

    rubric = vf.Rubric(parser=parser, funcs=[final_answer], weights=[1.0])
    return OPSDRLMHarnessOrientationEnv(dataset=dataset, parser=parser, rubric=rubric)
