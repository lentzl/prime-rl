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
        "capability": "General terminal file reading and straightforward Python calculations.",
        "limitations": "No specialization for multi-artifact reconciliation or source structure.",
    },
    "table_analyst": {
        "capability": "CSV and JSON joins, filters, grouping, reconciliation, and exact integer arithmetic.",
        "limitations": "Not specialized for Python AST or source-configuration inspection.",
    },
    "source_inspector": {
        "capability": "Python AST and source/configuration inspection with exact structural calculations.",
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
HARNESS_ASSIGNMENTS = {
    "generic_worker": (
        "Read {root}/values.json as a top-level JSON integer list. Compute sum((index + 1) * value for index, value in enumerate(values)). Send exactly one compact JSON object with integer key `value` to receiver_role='parent', then stop.",
        "Load the complete integer list from {root}/numbers.json with Python. Compute its exact index-weighted checksum using one-based indices. Send exactly one compact JSON object with integer key `value` to receiver_role='parent', then stop.",
        "Inspect {root}/measurements.json, which is one plain JSON list of integers. Multiply each integer by its one-based position and sum the products exactly. Send exactly one compact JSON object with integer key `value` to receiver_role='parent', then stop.",
        "Read every integer in the top-level list at {root}/payload.json and calculate the exact weighted total sum((index + 1) * value). Send exactly one compact JSON object with integer key `value` to receiver_role='parent', then stop.",
    ),
    "table_analyst": (
        "Read {root}/ledger.csv and {root}/rates.json. Join every CSV row to its JSON account rate, keep active rows, and sum quantity * rate as an exact integer. Send exactly one compact JSON object with integer key `value` to receiver_role='parent', then stop.",
        "Read {root}/inventory.csv and {root}/corrections.json. For every SKU compute opening + received - shipped + its JSON correction, then sum the reconciled quantities across all SKUs. Send exactly one compact JSON object with integer key `value` to receiver_role='parent', then stop.",
        "Join {root}/transactions.csv with {root}/multipliers.json by customer, filter to rows whose status is active, group duplicate customers, and compute the exact integer weighted total. Send exactly one compact JSON object with integer key `value` to receiver_role='parent', then stop.",
        "Reconcile duplicate product rows in {root}/stock.csv against the integer adjustments in {root}/adjustments.json, then return the exact aggregate corrected stock. Send exactly one compact JSON object with integer key `value` to receiver_role='parent', then stop.",
    ),
    "source_inspector": (
        "Parse the complete Python files {root}/alpha.py and {root}/beta.py with ast. Count every FunctionDef, AsyncFunctionDef, and function node with at least one decorator, then compute 2 * FunctionDef + 3 * AsyncFunctionDef + decorated_function_nodes. Send exactly one compact JSON object with integer key `value` to receiver_role='parent', then stop.",
        "Read {root}/service.toml with tomllib and {root}/features.env as KEY=value lines. Compute runtime workers * timeout_seconds + the number of feature values exactly equal to `true`. Send exactly one compact JSON object with integer key `value` to receiver_role='parent', then stop.",
        "Use Python ast on all of {root}/handlers.py and {root}/routes.py. Count synchronous definitions, asynchronous definitions, and decorated definitions and calculate the requested exact structural score. Send exactly one compact JSON object with integer key `value` to receiver_role='parent', then stop.",
        "Inspect {root}/runtime.toml and {root}/flags.env using tomllib and exact KEY=value parsing. Combine the configured worker and timeout integers with the count of enabled true flags. Send exactly one compact JSON object with integer key `value` to receiver_role='parent', then stop.",
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
    assignment_variant: int | None


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
    assignment_style: Literal["abstract", "harness_shaped"] = "abstract"
    registry_mode: Literal["permuted", "fixed"] = "permuted"


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
            if self.config.registry_mode == "fixed":
                permutation = EXPERT_IDS
            else:
                permutation = permutations[
                    (permutation_round + split_offset) % len(permutations)
                ]
            profile_by_expert_id = dict(zip(EXPERT_IDS, permutation, strict=True))
            answer = next(
                expert_id
                for expert_id, profile_id in profile_by_expert_id.items()
                if profile_id == required_profile
            )
            assignment_variant = None
            if self.config.assignment_style == "harness_shaped":
                assignment_templates = HARNESS_ASSIGNMENTS[required_profile]
                assignment_variant = (
                    permutation_round
                    if self.config.registry_mode == "fixed"
                    else permutation_round // len(permutations)
                ) % len(assignment_templates)
                assignment = assignment_templates[assignment_variant].format(
                    root=(
                        "/workspace/specialist-worker/"
                        f"v{assignment_variant}-"
                        f"i{self.config.start_index + position}"
                    )
                )
            else:
                assignment = ASSIGNMENTS[required_profile]
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
                    {"objective": assignment},
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
                        assignment_variant=assignment_variant,
                    ),
                    self.config.task,
                )
            )
        return tasks
