from types import SimpleNamespace

import pytest

from prime_rl.configs.orchestrator import TrainEnvConfig
from prime_rl.orchestrator.train_source import TrainSource


def _env(*, num_tasks: int | None, task_indices: list[int] | None = None):
    return SimpleNamespace(
        name="test-env",
        num_tasks=num_tasks,
        requires_group_scoring=False,
        config=SimpleNamespace(ratio=1.0, group_size=1, task_indices=task_indices),
    )


def test_train_env_task_indices_must_be_nonempty_unique_nonnegative_integers():
    assert TrainEnvConfig(id="test", task_indices=[2, 0]).task_indices == [2, 0]

    for task_indices in ([], [0, 0], [-1], [True], ["1"]):
        with pytest.raises(ValueError):
            TrainEnvConfig(id="test", task_indices=task_indices)


def test_finite_task_subset_repeats_only_selected_indices():
    source = TrainSource([_env(num_tasks=5, task_indices=[1, 3])], seed=0)

    examples = [source.next_example(available_permits=1) for _ in range(6)]

    assert all(example is not None for example in examples)
    assert [example["task_idx"] for example in examples if example is not None].count(1) == 3
    assert [example["task_idx"] for example in examples if example is not None].count(3) == 3
    assert {example["env_name"] for example in examples if example is not None} == {"test-env"}


def test_finite_task_subset_rejects_out_of_range_indices():
    with pytest.raises(ValueError, match=r"task_indices \[4\] exceed taskset size 4"):
        TrainSource([_env(num_tasks=4, task_indices=[0, 4])], seed=0)


def test_infinite_taskset_rejects_task_indices():
    with pytest.raises(ValueError, match="infinite taskset"):
        TrainSource([_env(num_tasks=None, task_indices=[0])], seed=0)


def test_finite_taskset_defaults_to_all_indices():
    source = TrainSource([_env(num_tasks=3)], seed=0)

    examples = [source.next_example(available_permits=1) for _ in range(3)]

    assert {example["task_idx"] for example in examples if example is not None} == {0, 1, 2}
