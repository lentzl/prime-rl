"""Outcome-scored specialist routing from public capability metadata."""

from __future__ import annotations

import itertools
import json
import re
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

EXPERT_IDS = ("generic_worker", "table_analyst", "source_inspector")
PROFILES = {
    "generic_worker": {
        "capability": "Plain file reading and straightforward Python calculations.",
        "limitations": "No specialization for table reconciliation or source structure.",
    },
    "table_analyst": {
        "capability": "CSV and JSON joins, filters, grouping, and exact reconciliation.",
        "limitations": "Not specialized for Python AST or source configuration inspection.",
    },
    "source_inspector": {
        "capability": "Python AST and source/configuration inspection with exact counts.",
        "limitations": "Not specialized for tabular joins or ledger reconciliation.",
    },
}
ASSIGNMENTS = {
    "generic_worker": (
        "Read one plain JSON list of integers and compute its exact weighted sum."
    ),
    "table_analyst": (
        "Join a CSV ledger to JSON rates by account, reconcile duplicate rows, and compute "
        "the exact filtered integer total."
    ),
    "source_inspector": (
        "Parse Python modules with ast, inspect a TOML service configuration, and compute "
        "the requested exact structural count."
    ),
}
SYSTEM = """You are a terminal-expert routing policy.
The cognitive action is already delegate_terminal. Read the public capability registry
and assignment, then select the cheapest sufficient registered worker. Return exactly
one JSON object with one string field named expert_id. Do not solve the assignment."""
_JSON_OBJECT = re.compile(r"\{[^{}]+\}")


def selected_expert(reply: str | None) -> str | None:
    if not isinstance(reply, str):
        return None
    matches = []
    for candidate in _JSON_OBJECT.findall(reply):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(parsed, dict)
            and set(parsed) == {"expert_id"}
            and parsed["expert_id"] in EXPERT_IDS
        ):
            matches.append(parsed["expert_id"])
    return matches[0] if len(matches) == 1 else None


class SpecialistRouterData(vf.TaskData):
    answer: str
    required_profile: str
    profile_by_expert_id: dict[str, str]


class SpecialistRouterTask(vf.Task[SpecialistRouterData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def routing_outcome(self, trace: vf.Trace) -> float:
        return float(selected_expert(trace.last_reply) == self.data.answer)

    @vf.reward(weight=0.1)
    async def valid_transport(self, trace: vf.Trace) -> float:
        return float(selected_expert(trace.last_reply) is not None)

    @vf.metric
    async def selected_identity(self, trace: vf.Trace) -> dict[str, float]:
        selected = selected_expert(trace.last_reply)
        return {
            f"selected_{expert_id}": float(selected == expert_id)
            for expert_id in EXPERT_IDS
        }


class SpecialistRouterConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    count: int = Field(default=96, ge=6)
    start_index: int = 39000
    required_profile: Literal[
        "generic_worker", "table_analyst", "source_inspector"
    ] | None = None


class SpecialistRouterTaskset(
    vf.Taskset[SpecialistRouterTask, SpecialistRouterConfig]
):
    def load(self) -> list[SpecialistRouterTask]:
        permutations = list(itertools.permutations(EXPERT_IDS))
        tasks = []
        split_offset = 0 if self.config.split == "train" else 10_000
        for position in range(self.config.count):
            required_profile = (
                self.config.required_profile
                or EXPERT_IDS[position % len(EXPERT_IDS)]
            )
            permutation_round = (
                position
                if self.config.required_profile is not None
                else position // len(EXPERT_IDS)
            )
            permutation = permutations[
                (permutation_round + split_offset) % len(permutations)
            ]
            profile_by_expert_id = dict(zip(EXPERT_IDS, permutation, strict=True))
            answer = next(
                expert_id
                for expert_id, profile_id in profile_by_expert_id.items()
                if profile_id == required_profile
            )
            registry = []
            for expert_id in EXPERT_IDS:
                profile = PROFILES[profile_by_expert_id[expert_id]]
                registry.append(
                    json.dumps(
                        {
                            "expert_id": expert_id,
                            "role": "terminal_worker",
                            "capability": profile["capability"],
                            "limitations": profile["limitations"],
                            "relative_cost": 1.0,
                        },
                        separators=(",", ":"),
                    )
                )
            prompt = (
                "[capability registry]\n"
                + "\n".join(registry)
                + "\n[terminal specialist assignment]\n"
                + json.dumps(
                    {"objective": ASSIGNMENTS[required_profile]},
                    separators=(",", ":"),
                )
            )
            tasks.append(
                SpecialistRouterTask(
                    SpecialistRouterData(
                        idx=self.config.start_index + position,
                        prompt=prompt,
                        system_prompt=SYSTEM,
                        answer=answer,
                        required_profile=required_profile,
                        profile_by_expert_id=profile_by_expert_id,
                    ),
                    self.config.task,
                )
            )
        return tasks
