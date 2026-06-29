"""RLM-shaped multi-turn env for OPSD feedback-scoring smoke runs.

This fixture is still tiny and deterministic, but its episode shape is closer
to the RLM harness loop than the echo smoke:

1. the model proposes or uses a reusable helper;
2. the environment returns execution-style feedback;
3. the model repairs/uses the helper and submits a final answer.

The example's ``answer`` field is a demonstration consumed by OPSD as the
feedback-conditioned reference context.
"""

from datasets import Dataset

import verifiers as vf
from verifiers.types import Messages, State

TASK = """You are operating inside a tiny REPL harness.

Available persistent library card:
- normalize_tokens(text): lowercase text, keep alphanumeric characters, split
  on whitespace.

Task:
Use or reconstruct that helper to normalize this noisy record:

    "Alpha!! beta, ALPHA... gamma?"

When you are ready, submit exactly:
<answer>alpha beta alpha gamma</answer>
"""

DIAGNOSTIC_FEEDBACK = """Execution feedback:
- A reusable helper should own the normalization rule.
- The previous attempt will be accepted only if the final answer is wrapped in
  <answer> tags.
- Continue from the feedback; do not restart the whole episode.
"""

RECOVERY_PROMPT = (
    "Now repair the attempt using the helper behavior and submit the final answer."
)

DEMONSTRATION = """A good harness-style solution externalizes the text rule into a helper,
then uses feedback to repair the final submission:
1. inspect the requested helper behavior;
2. normalize the noisy record to alpha beta alpha gamma;
3. submit <answer>alpha beta alpha gamma</answer>.
"""


class OPSDRLMHarnessSmokeEnv(vf.MultiTurnEnv):
    @vf.stop
    async def final_feedback_seen(self, state: State) -> bool:
        return len(state["trajectory"]) >= 3

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        turn = len(state["trajectory"])
        if turn == 1:
            return [vf.UserMessage(content=DIAGNOSTIC_FEEDBACK)]
        return [vf.UserMessage(content=RECOVERY_PROMPT)]


def load_environment() -> vf.Environment:
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": TASK}],
                "answer": DEMONSTRATION,
                "info": {"expected": "alpha beta alpha gamma"},
            }
        ]
    )
    parser = vf.XMLParser(["answer"], answer_field="answer")

    def final_answer(parser, completion, info, **kwargs) -> float:
        parsed = parser.parse_answer(completion) or ""
        expected = info["expected"]
        return 1.0 if " ".join(parsed.lower().split()) == expected else 0.0

    rubric = vf.Rubric(parser=parser, funcs=[final_answer], weights=[1.0])
    return OPSDRLMHarnessSmokeEnv(dataset=dataset, parser=parser, rubric=rubric)
