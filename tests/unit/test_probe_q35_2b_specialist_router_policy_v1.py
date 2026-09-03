import importlib.util
import sys
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[2]
        / "scripts/probe_q35_2b_specialist_router_policy_v1.py"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_router_probe_only_maps_terminal_root_families() -> None:
    module = _module()
    expected = {
        "specialist_local": None,
        "specialist_generic": "generic_worker",
        "specialist_table_join": "table_analyst",
        "specialist_table_reconcile": "table_analyst",
        "specialist_source_ast": "source_inspector",
        "specialist_source_config": "source_inspector",
        "specialist_recursive_table": None,
        "specialist_recursive_source": None,
    }
    assert {
        family: module._root_expert(family) for family in module.SPECIALIST_FAMILIES
    } == expected


def test_router_probe_screens_are_frozen_and_disjoint() -> None:
    module = _module()
    assert module.FROZEN_SCREENS == {
        37700: 20261209,
        37800: 20261210,
        38000: 20261212,
        38200: 20261213,
        38300: 20261214,
        38400: 20261215,
        38500: 20261216,
        38600: 20261217,
        38700: 20261218,
    }
    assert not set(module.FROZEN_SCREENS) & {
        35100,
        37100,
        37200,
        37300,
        37400,
        37500,
        37600,
        37900,
    }
