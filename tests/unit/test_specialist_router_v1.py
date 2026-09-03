import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[2]
        / "environments/specialist_router_v1/specialist_router_v1/taskset.py"
    )
    spec = importlib.util.spec_from_file_location("specialist_router_v1_taskset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_router_taskset_balances_profiles_targets_and_permutations() -> None:
    module = _module()
    taskset = module.SpecialistRouterTaskset(module.SpecialistRouterConfig(count=96))
    tasks = taskset.load()

    assert Counter(task.data.required_profile for task in tasks) == Counter(
        {expert_id: 32 for expert_id in module.EXPERT_IDS}
    )
    assert Counter(task.data.answer for task in tasks) == Counter(
        {expert_id: 32 for expert_id in module.EXPERT_IDS}
    )
    mappings = defaultdict(set)
    for task in tasks:
        mappings[task.data.required_profile].add(
            tuple(sorted(task.data.profile_by_expert_id.items()))
        )
        assert (
            task.data.profile_by_expert_id[task.data.answer]
            == task.data.required_profile
        )
    assert {key: len(values) for key, values in mappings.items()} == {
        expert_id: 6 for expert_id in module.EXPERT_IDS
    }


def test_selected_expert_requires_one_valid_exact_object() -> None:
    module = _module()
    assert module.selected_expert('{"expert_id":"source_inspector"}') == (
        "source_inspector"
    )
    assert module.selected_expert('{"expert_id":"unknown"}') is None
    assert module.selected_expert('{"expert_id":"table_analyst","extra":1}') is None
    assert (
        module.selected_expert(
            '{"expert_id":"generic_worker"} {"expert_id":"table_analyst"}'
        )
        is None
    )


def test_router_taskset_can_pin_one_required_profile_without_pinning_identity() -> None:
    module = _module()
    tasks = module.SpecialistRouterTaskset(
        module.SpecialistRouterConfig(
            count=18,
            required_profile="source_inspector",
            start_index=39100,
        )
    ).load()

    assert {task.data.required_profile for task in tasks} == {"source_inspector"}
    assert Counter(task.data.answer for task in tasks) == Counter(
        {expert_id: 6 for expert_id in module.EXPERT_IDS}
    )
    assert len(
        {
            tuple(sorted(task.data.profile_by_expert_id.items()))
            for task in tasks
        }
    ) == 6


def test_router_taskset_can_use_diverse_harness_shaped_assignments() -> None:
    module = _module()
    tasks = module.SpecialistRouterTaskset(
        module.SpecialistRouterConfig(
            count=24,
            required_profile="table_analyst",
            start_index=39400,
            assignment_style="harness_shaped",
        )
    ).load()

    assert Counter(task.data.assignment_variant for task in tasks) == Counter(
        {variant: 6 for variant in range(4)}
    )
    mappings_by_variant = defaultdict(set)
    for task in tasks:
        mappings_by_variant[task.data.assignment_variant].add(
            tuple(sorted(task.data.profile_by_expert_id.items()))
        )
    assert {variant: len(mappings) for variant, mappings in mappings_by_variant.items()} == {
        variant: 6 for variant in range(4)
    }
    assert all("/workspace/specialist-worker/" in task.data.prompt for task in tasks)
    assert all("receiver_role='parent'" in task.data.prompt for task in tasks)
    assert Counter(task.data.answer for task in tasks) == Counter(
        {expert_id: 8 for expert_id in module.EXPERT_IDS}
    )


def test_router_taskset_can_leak_stable_live_registry_identities() -> None:
    module = _module()
    tasks = module.SpecialistRouterTaskset(
        module.SpecialistRouterConfig(
            count=24,
            required_profile="table_analyst",
            assignment_style="harness_shaped",
            registry_mode="fixed",
        )
    ).load()

    expected_mapping = {expert_id: expert_id for expert_id in module.EXPERT_IDS}
    assert {task.data.answer for task in tasks} == {"table_analyst"}
    assert all(task.data.profile_by_expert_id == expected_mapping for task in tasks)
    assert Counter(task.data.assignment_variant for task in tasks) == Counter(
        {variant: 6 for variant in range(4)}
    )
