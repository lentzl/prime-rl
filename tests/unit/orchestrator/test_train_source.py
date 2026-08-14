from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.train_source import TrainSource


def make_envs():
    return [
        SimpleNamespace(
            name="native",
            config=SimpleNamespace(ratio=2.0),
            tasks=iter(range(64)),
            num_tasks=64,
        ),
        SimpleNamespace(
            name="diagnostic",
            config=SimpleNamespace(ratio=1.0),
            tasks=iter(range(64)),
            num_tasks=64,
        ),
    ]


def test_weighted_round_robin_produces_exact_complete_cycles() -> None:
    source = TrainSource(make_envs(), source_selection="weighted_round_robin")

    selected = [source.next_example()["env_name"] for _ in range(12)]

    assert selected == ["native", "diagnostic", "native"] * 4


def test_weighted_round_robin_state_resumes_exactly() -> None:
    source = TrainSource(make_envs(), source_selection="weighted_round_robin")
    for _ in range(5):
        source.next_example()
    state = source.state_dict()
    expected = [source.next_example()["env_name"] for _ in range(12)]

    resumed = TrainSource(make_envs(), source_selection="weighted_round_robin")
    resumed.load_state_dict(state)

    assert [resumed.next_example()["env_name"] for _ in range(12)] == expected


def test_source_selection_must_match_checkpoint() -> None:
    source = TrainSource(make_envs(), source_selection="weighted_round_robin")
    checkpoint = source.state_dict()
    resumed = TrainSource(make_envs())

    with pytest.raises(ValueError, match="differs from the checkpoint"):
        resumed.load_state_dict(checkpoint)
