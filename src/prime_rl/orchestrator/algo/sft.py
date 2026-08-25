from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import SFTAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.utils.utils import import_object

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
        self.filter_fn: Callable[..., list[list[bool]]] | None = None
        if config.filter is not None:
            self.filter_fn = partial(import_object(config.filter.import_path), **config.filter.kwargs)

    async def score_rollout(self, rollout: Rollout) -> None:
        if self.sampled_session_scope != "all":
            retain_session_scope_samples(
                rollout,
                scope=self.sampled_session_scope,
                drop_empty=False,
            )
        if self.filter_fn is not None:
            retain_filtered_samples(rollout, self.filter_fn)
        if self.sampled_session_scope != "all" or self.filter_fn is not None:
            rollout.samples = [sample for sample in rollout.samples if any(sample.mask)]


def retain_filtered_samples(
    rollout: Rollout,
    filter_fn: Callable[..., list[list[bool]]],
) -> None:
    """Narrow sampled SFT tokens with one branch-aligned environment mask."""
    trainable = list(iter_trainable_branches(rollout))
    if len(trainable) != len(rollout.samples):
        raise ValueError(
            "SFT filter branch/sample mismatch: "
            f"{len(trainable)} branches, {len(rollout.samples)} samples"
        )
    masks = filter_fn(rollout)
    if not isinstance(masks, list) or len(masks) != len(trainable):
        got = len(masks) if isinstance(masks, list) else type(masks).__name__
        raise ValueError(
            "SFT filter must return one keep-mask per trainable branch: "
            f"got {got}, expected {len(trainable)}"
        )

    for branch_index, (sample, (branch, branch_mask), keep_mask) in enumerate(
        zip(rollout.samples, trainable, masks, strict=True)
    ):
        if list(sample.token_ids) != list(branch.token_ids):
            raise ValueError(f"SFT filter sample {branch_index} tokens do not match its trace branch")
        expected = len(branch.token_ids)
        if not isinstance(keep_mask, list) or len(keep_mask) != expected:
            got = len(keep_mask) if isinstance(keep_mask, list) else type(keep_mask).__name__
            raise ValueError(
                f"SFT filter mask for branch {branch_index} must span the branch's tokens: "
                f"got {got}, expected {expected}"
            )
        if any(type(value) is not bool for value in keep_mask):
            raise ValueError(f"SFT filter mask for branch {branch_index} must contain booleans")
        if len(sample.mask) != len(branch_mask) or any(
            selected and not trainable_token
            for selected, trainable_token in zip(sample.mask, branch_mask, strict=True)
        ):
            raise ValueError(f"SFT filter sample {branch_index} mask exceeds its trace branch")
        sample.mask = [
            selected and keep
            for selected, keep in zip(sample.mask, keep_mask, strict=True)
        ]


def retain_session_scope_samples(
    rollout: Rollout,
    *,
    scope: str,
    drop_empty: bool = True,
) -> None:
    """Keep sampled tokens inside or outside the primary client session.

    Agent harnesses may place coordinator and child calls in one trace. The
    primary root is the first root node. Model calls in client-session graphs
    must have explicit, unambiguous lineage. Wholly unscoped auxiliary graphs
    (for example, harness refinement) are excluded from both role scopes.
    """
    if scope not in {"root", "non_root"}:
        raise ValueError(f"session-scoped training does not support scope {scope!r}")
    if not rollout.nodes:
        raise ValueError("session-scoped training requires a non-empty trace graph")
    root_index = next(
        (index for index, node in enumerate(rollout.nodes) if node.parent is None),
        None,
    )
    if root_index is None:
        raise ValueError("session-scoped training requires a graph root")

    def graph_root(node_index: int) -> int:
        seen: set[int] = set()
        while True:
            if node_index in seen:
                raise ValueError("session-scoped training found a cycle in the trace graph")
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

    lineage_by_graph_root: dict[int, set[str | None]] = {}
    for node_index, lineage in lineage_by_node.items():
        lineage_by_graph_root.setdefault(graph_root(node_index), set()).update(lineage)

    def session_for_node(node_index: int) -> str | None:
        lineage = lineage_by_node.get(node_index, set())
        valid_lineage = {session_id for session_id in lineage if session_id is not None}
        if len(lineage) == 1 and len(valid_lineage) == 1:
            return next(iter(valid_lineage))
        node_graph_root = graph_root(node_index)
        if (
            node_graph_root != root_index
            and lineage_by_graph_root.get(node_graph_root) == {None}
        ):
            return None
        raise ValueError(
            "session-scoped training requires exactly one client session for every "
            f"model-call node (node {node_index} has {len(valid_lineage)} valid candidate(s))"
        )

    root_sessions = {
        session_for_node(node_index)
        for node_index in lineage_by_node
        if graph_root(node_index) == root_index
    }
    if len(root_sessions) != 1:
        raise ValueError(
            "session-scoped training requires exactly one client session on the primary "
            f"graph root, found {len(root_sessions)}"
        )
    root_session_id = next(iter(root_sessions))
    assert root_session_id is not None

    node_indices = {id(node): index for index, node in enumerate(rollout.nodes)}
    trainable = list(iter_trainable_branches(rollout))
    if len(trainable) != len(rollout.samples):
        raise ValueError(
            "session-scoped training branch/sample mismatch: "
            f"{len(trainable)} branches, {len(rollout.samples)} samples"
        )

    for sample, (branch, branch_mask) in zip(rollout.samples, trainable):
        if list(sample.mask) != branch_mask:
            raise ValueError("session-scoped training sample mask does not match its trace branch")
        offset = 0
        for node in branch.nodes:
            span = len(node.token_ids)
            if node.sampled and any(branch_mask[offset : offset + span]):
                node_index = node_indices[id(node)]
                session_id = session_for_node(node_index)
                selected = session_id is not None and session_id == root_session_id
                if scope == "non_root":
                    selected = session_id is not None and session_id != root_session_id
                if not selected:
                    sample.mask[offset : offset + span] = [False] * span
            offset += span
    if drop_empty:
        rollout.samples = [sample for sample in rollout.samples if any(sample.mask)]


def retain_root_session_samples(rollout: Rollout, *, drop_empty: bool = True) -> None:
    """Backward-compatible wrapper retaining only primary-session tokens."""
    retain_session_scope_samples(rollout, scope="root", drop_empty=drop_empty)
