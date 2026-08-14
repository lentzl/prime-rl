import json

import pytest

from scripts.generate_programmatic_episodic_memory_v2 import (
    input_fingerprint,
    validate_split_disjointness,
)


def _row(prompt: str, history: str) -> dict:
    return {
        "messages_json": json.dumps(
            [
                {"role": "system", "content": "Use IPython."},
                {"role": "user", "content": prompt},
            ]
        ),
        "tools": json.dumps([{"type": "function", "function": {"name": "ipython"}}]),
        "workspace_files_json": json.dumps({"/workspace/journal.log": history}),
    }


def test_full_input_fingerprint_includes_workspace_state() -> None:
    first = _row("Read the current state.", "status=old\n")
    second = _row("Read the current state.", "status=new\n")

    assert input_fingerprint(first) != input_fingerprint(second)
    validate_split_disjointness({"train": [first], "familiar_heldout": [second]})


def test_split_validation_rejects_exact_model_input_overlap() -> None:
    row = _row("Read the current state.", "status=old\n")

    with pytest.raises(AssertionError, match="full-input overlap"):
        validate_split_disjointness({"train": [row], "familiar_heldout": [row.copy()]})
