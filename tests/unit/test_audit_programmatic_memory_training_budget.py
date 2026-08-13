import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "audit_programmatic_memory_training_budget.py"
SPEC = importlib.util.spec_from_file_location("audit_programmatic_memory_training_budget", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_packed_step_budget_accounts_for_padding_and_epochs() -> None:
    budget = MODULE.packed_step_budget(
        [6, 6, 4],
        epochs=2,
        seq_len=10,
        batch_size=2,
    )

    assert budget == {
        "epochs": 2,
        "packs": 4,
        "steps": 2,
        "rendered_tokens": 32,
        "padded_tokens": 40,
        "packing_utilization": 0.8,
    }


def test_packed_step_budget_rejects_truncation() -> None:
    with pytest.raises(ValueError, match="exceeds SFT seq_len"):
        MODULE.packed_step_budget([11], epochs=1, seq_len=10, batch_size=1)


def test_percentile_uses_nearest_rank() -> None:
    assert MODULE.percentile([1, 2, 3, 4, 5], 0.95) == 5
