import importlib.util
from pathlib import Path

from procedural_harness_master_v1.taskset import (
    ProceduralHarnessMasterConfig,
    ProceduralHarnessMasterTaskset,
)


def _module():
    path = Path(__file__).parents[2] / "scripts" / "build_q35_2b_spade_rung0_hints_v1.py"
    spec = importlib.util.spec_from_file_location(
        "build_q35_2b_spade_rung0_hints_v1", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task():
    return ProceduralHarnessMasterTaskset(
        ProceduralHarnessMasterConfig(
            count=1,
            start_index=4_006_000,
            master_seed=20260819,
            curriculum_rung="natural_n1a_local",
            private_payload_mode="finding_card",
        )
    ).load()[0]


def test_hint_quality_gate_requires_operational_async_contract() -> None:
    module = _module()
    task = _task()
    child = task.data.oracle["children"][0]
    local_path = next(
        path
        for path, ownership in task.data.oracle["resource_ownership"].items()
        if ownership["owner"] == "coordinator"
    )
    shallow = (
        f"Spawn {child['name']}, retain its handle, do local work, yield, and answer."
    )
    operational = (
        "Create exactly one child with handle = await rlm(\"Review the child-owned evidence "
        "and call await agent_message.send(..., receiver_role='parent') exactly once\", "
        f"name={child['name']!r}) and retain the handle. "
        "Tell that child to reply exactly once using await agent_message.send(..., "
        "receiver_role='parent'). The coordinator must not read or inspect the child-owned "
        f"path {child['resource_path']}. Read only the coordinator-owned path {local_path} "
        "and perform its specified local operation in IPython. Do not poll, sleep, discover "
        "agents, await the child, or spawn a replacement. After local work, stop tool use "
        "and yield by ending the coordinator turn briefly. Resume only from the delivered "
        "child report, compute the requested fields, and return bare JSON."
    )

    assert module._hint_contract_gaps(task, shallow)
    assert module._hint_contract_gaps(task, operational) == []
    assert module._leaks_protected_value(
        operational, module._protected_values(task)
    ) == []


def test_designer_view_omits_answer_and_private_payload_values() -> None:
    module = _module()
    task = _task()
    view = module._canonical_json(module._designer_view(task))

    assert '"final_answer_keys"' in view
    assert '"final_answer":{' not in view
    assert '"private_resources"' not in view
    assert str(task.data.oracle["final_answer"]["finding"]) not in view
