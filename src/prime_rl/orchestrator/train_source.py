"""TrainSource: weighted round-robin across train envs, infinite pull.

Weights are each env's configured ``ratio`` (default 1, i.e. equal weight
per env). A v1 env serves the tasks the orchestrator loaded client-side: a
finite one as a shuffled table (reshuffled on cursor exhaustion), an
infinite one (``num_tasks is None``) straight off its generator — every
pull is a fresh task and there are no epochs to shuffle. A legacy env's
dataset lives on its server, so it serves shuffled task *indices*."""

from __future__ import annotations

import random
from collections.abc import Iterator

import verifiers.v1 as vf

from prime_rl.orchestrator.envs import TrainEnvs


class TrainSource:
    """``next_example(available_permits)`` picks a weighted-RR env and
    returns its next example (or ``None`` when the env's per-call permit
    cost doesn't fit — the dispatch loop retries when permits free up).
    Returned dicts carry ``env_name`` + ``task_idx`` (+ ``task`` for v1 envs,
    whose data is shipped to the env server at dispatch)."""

    def __init__(self, train_envs: TrainEnvs, *, seed: int | None) -> None:
        self.rng = random.Random(seed)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        # A finite env's shuffled example table; ``None`` for an infinite env,
        # whose generator (``self.iters``) is pulled per example.
        self.examples: dict[str, list[dict] | None] = {}
        self.iters: dict[str, Iterator[vf.Task]] = {}
        self.cursors: dict[str, int] = {}
        # Group-scoring envs reserve ``group_size`` permits up front;
        # per-rollout envs need 1
        self.env_costs: dict[str, int] = {}
        for env in self.envs:
            tasks = getattr(env, "tasks", None)
            task_indices = env.config.task_indices
            if tasks is None:  # legacy: sample over the index range from info()
                if env.num_tasks is None:
                    if task_indices is not None:
                        raise ValueError(f"Train env {env.name} has an infinite taskset and cannot select task_indices")
                    raise ValueError(f"Legacy train env {env.name} must report a finite taskset size")
                selected = list(range(env.num_tasks)) if task_indices is None else task_indices
                out_of_range = [index for index in selected if index >= env.num_tasks]
                if out_of_range:
                    raise ValueError(
                        f"Train env {env.name} task_indices {out_of_range} exceed taskset size {env.num_tasks}"
                    )
                rows: list[dict] = [{"task_idx": index, "env_name": env.name} for index in selected]
                self.rng.shuffle(rows)
                self.examples[env.name] = rows
            elif env.num_tasks is None:  # infinite: pull the generator per example
                if task_indices is not None:
                    raise ValueError(f"Train env {env.name} has an infinite taskset and cannot select task_indices")
                self.examples[env.name] = None
                self.iters[env.name] = tasks
            else:
                task_by_index = {task.data.idx: task for task in tasks}
                selected = list(task_by_index) if task_indices is None else task_indices
                missing = [index for index in selected if index not in task_by_index]
                if missing:
                    raise ValueError(
                        f"Train env {env.name} task_indices {missing} are not present in the loaded taskset"
                    )
                rows = [{"task_idx": index, "task": task_by_index[index], "env_name": env.name} for index in selected]
                self.rng.shuffle(rows)
                self.examples[env.name] = rows
            self.cursors[env.name] = 0
            self.env_costs[env.name] = env.config.group_size if env.requires_group_scoring else 1

        self.env_names = [e.name for e in self.envs]
        self.weights: list[float] = [float(e.config.ratio) for e in self.envs]

    def next_example(self, available_permits: int) -> dict | None:
        env_name = self.rng.choices(self.env_names, weights=self.weights, k=1)[0]
        if self.env_costs[env_name] > available_permits:
            return None
        rows = self.examples[env_name]
        cursor = self.cursors[env_name]
        if rows is None:  # infinite env: pull the next generated task
            task = next(self.iters[env_name])
            return {"task_idx": task.data.idx, "task": task, "env_name": env_name}
        if cursor >= len(rows):
            self.rng.shuffle(rows)
            cursor = 0
        example = rows[cursor]
        self.cursors[env_name] = cursor + 1
        return example
