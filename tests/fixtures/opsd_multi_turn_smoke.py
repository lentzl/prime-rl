"""Tiny multi-turn env for OPSD reference-scoring smoke runs.

The env is intentionally boring: two user turns, two assistant turns, and a
static demonstration in the example's ``answer`` field. It exists to exercise
the Prime rollout -> OPSD multi-turn scoring -> trainer transport path without
depending on an external dataset.
"""

from datasets import Dataset

import verifiers as vf
from verifiers.types import Messages, State

PHRASES = ["say alpha", "say beta"]
DEMONSTRATION = "For each user message, answer with the requested phrase exactly."


class OPSDMultiTurnSmokeEnv(vf.MultiTurnEnv):
    @vf.stop
    async def all_turns_seen(self, state: State) -> bool:
        return len(state["trajectory"]) >= len(state["info"]["phrases"])

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        next_phrase = state["info"]["phrases"][len(state["trajectory"])]
        return [vf.UserMessage(content=next_phrase)]


def load_environment(phrases: list[str] = PHRASES) -> vf.Environment:
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": phrases[0]}],
                "answer": DEMONSTRATION,
                "info": {"phrases": phrases},
            }
        ]
    )
    parser = vf.Parser()

    def echoed(parser, completion, info, **kwargs) -> float:
        replies = [m["content"] for m in parser.get_assistant_messages(completion)]
        phrases = info["phrases"]
        if len(replies) < len(phrases):
            return 0.0
        matched = sum(phrase.lower() in (reply or "").lower() for reply, phrase in zip(replies, phrases))
        return matched / len(phrases)

    rubric = vf.Rubric(parser=parser, funcs=[echoed], weights=[1.0])
    return OPSDMultiTurnSmokeEnv(dataset=dataset, parser=parser, rubric=rubric)
