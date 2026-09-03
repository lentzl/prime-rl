import importlib.util
import json
import sys
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[2]
        / "scripts/probe_q35_2b_split_specialist_routing_v1.py"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_payloads_force_exactly_one_field() -> None:
    module = _module()
    messages = [{"role": "user", "content": "route this"}]

    action = module._action_payload(model="e33", messages=messages, seed=7)
    action_function = action["tools"][0]["function"]
    assert action["tool_choice"]["function"]["name"] == "select_cognitive_action"
    assert action_function["parameters"]["required"] == ["action"]
    assert set(action_function["parameters"]["properties"]) == {"action"}
    assert action["parallel_tool_calls"] is False

    expert = module._expert_payload(model="e33", messages=messages, seed=8)
    expert_function = expert["tools"][0]["function"]
    assert expert["tool_choice"]["function"]["name"] == "select_expert"
    assert expert_function["parameters"]["required"] == ["expert_id"]
    assert set(expert_function["parameters"]["properties"]) == {"expert_id"}
    assert expert_function["parameters"]["properties"]["expert_id"]["enum"] == list(
        module.EXPERT_IDS
    )


def test_response_parser_distinguishes_typed_call_from_json_fallback() -> None:
    module = _module()
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "select_expert",
                                "arguments": json.dumps({"expert_id": "table_analyst"}),
                            },
                        }
                    ],
                }
            }
        ]
    }
    selected, exact, normalizable, _ = module._arguments_from_response(
        response, tool_name="select_expert", field="expert_id"
    )
    assert selected == "table_analyst"
    assert exact is True
    assert normalizable is True

    response["choices"][0]["message"] = {
        "role": "assistant",
        "content": json.dumps({"expert_id": "source_inspector"}),
    }
    selected, exact, normalizable, _ = module._arguments_from_response(
        response, tool_name="select_expert", field="expert_id"
    )
    assert selected == "source_inspector"
    assert exact is False
    assert normalizable is True


def test_expected_root_routes_cover_all_specialist_families() -> None:
    module = _module()
    expected = {
        "specialist_local": ("solve_owned", None),
        "specialist_generic": ("delegate_terminal", "generic_worker"),
        "specialist_table_join": ("delegate_terminal", "table_analyst"),
        "specialist_table_reconcile": ("delegate_terminal", "table_analyst"),
        "specialist_source_ast": ("delegate_terminal", "source_inspector"),
        "specialist_source_config": ("delegate_terminal", "source_inspector"),
        "specialist_recursive_table": ("delegate_coordinator", None),
        "specialist_recursive_source": ("delegate_coordinator", None),
    }
    assert {
        family: module._expected_root_route(family)
        for family in module.SPECIALIST_FAMILIES
    } == expected
