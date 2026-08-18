from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import SFTAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.trajectories import iter_trainable_branches

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class SFTDistillAlgorithm(Algorithm):
    """Hard distillation. Needs a teacher: the frozen model that generates the
    rollouts (``sampling.source``); the policy trains with CE on its tokens.

    Assigns no advantage — the ``ce`` loss ignores credit, and SFT trains on
    every sampled token. Reward-based filtering, if wanted, is an explicit
    filter, not smuggled through an unused advantage stream."""

    action_loss_type = "ce"

    def __init__(self, config: SFTAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.sampled_session_scope = config.sampled_session_scope

    async def score_rollout(self, rollout: Rollout) -> None:
        if self.sampled_session_scope == "root":
            retain_root_session_samples(rollout)


def retain_root_session_samples(rollout: Rollout) -> None:
    """Keep sampled tokens from the primary trace root's client session.

    Agent harnesses may place coordinator and child calls in one trace. The
    primary root is the first root node; every sampled node must have a model
    call with explicit client lineage. Ambiguous or missing lineage raises
    instead of silently training the wrong agent role.
    """
    if not rollout.nodes:
        raise ValueError("root-session SFT requires a non-empty trace graph")
    root_index = next(
        (index for index, node in enumerate(rollout.nodes) if node.parent is None),
        None,
    )
    if root_index is None:
        raise ValueError("root-session SFT requires a graph root")

    def graph_root(node_index: int) -> int:
        seen: set[int] = set()
        while True:
            if node_index in seen:
                raise ValueError("root-session SFT found a cycle in the trace graph")
            seen.add(node_index)
            parent = rollout.nodes[node_index].parent
            if parent is None:
                return node_index
            node_index = parent

    lineage_by_node: dict[int, set[str | None]] = {}
    for call in rollout.calls:
        if call.node is None:
            continue
        if not 0 <= call.node < len(rollout.nodes):
            raise ValueError(f"root-session SFT model call references invalid node {call.node}")
        lineage_by_node.setdefault(call.node, set()).add(call.client_session_id)

    def session_for_node(node_index: int) -> str:
        lineage = lineage_by_node.get(node_index, set())
        if len(lineage) != 1 or None in lineage:
            raise ValueError(
                "root-session SFT requires exactly one client session for every "
                f"model-call node (node {node_index} has {len(lineage)} valid candidate(s))"
            )
        session_id = next(iter(lineage))
        assert session_id is not None
        return session_id

    root_sessions = {
        session_for_node(node_index)
        for node_index in lineage_by_node
        if graph_root(node_index) == root_index
    }
    if len(root_sessions) != 1:
        raise ValueError(
            "root-session SFT requires exactly one client session on the primary "
            f"graph root, found {len(root_sessions)}"
        )
    root_session_id = next(iter(root_sessions))

    node_indices = {id(node): index for index, node in enumerate(rollout.nodes)}
    trainable = list(iter_trainable_branches(rollout))
    if len(trainable) != len(rollout.samples):
        raise ValueError(
            "root-session SFT branch/sample mismatch: "
            f"{len(trainable)} branches, {len(rollout.samples)} samples"
        )

    kept_samples = []
    for sample, (branch, branch_mask) in zip(rollout.samples, trainable):
        if list(sample.mask) != branch_mask:
            raise ValueError("root-session SFT sample mask does not match its trace branch")
        offset = 0
        for node in branch.nodes:
            span = len(node.token_ids)
            if node.sampled and any(branch_mask[offset : offset + span]):
                node_index = node_indices[id(node)]
                session_id = session_for_node(node_index)
                if session_id != root_session_id:
                    sample.mask[offset : offset + span] = [False] * span
            offset += span
        if any(sample.mask):
            kept_samples.append(sample)
    rollout.samples = kept_samples
