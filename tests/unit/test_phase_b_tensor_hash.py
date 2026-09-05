from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


class StateModule:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def state_dict(self) -> dict[str, Any]:
        return self._state


def _load_runner():
    repository = Path(__file__).resolve().parents[2]
    runner_path = repository / "scripts/latent/run_phase_b_fixed_depth_smoke_v1.py"
    spec = importlib.util.spec_from_file_location("phase_b_tensor_hash_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def _legacy_nonscalar_hash(module: StateModule) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        assert contiguous.dim() > 0
        digest.update(name.encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_module_hash_accepts_zero_dimensional_bfloat16_and_preserves_metadata() -> None:
    runner = _load_runner()
    scalar = StateModule({"output_scale": torch.tensor(1.5, dtype=torch.bfloat16)})
    length_one = StateModule({"output_scale": torch.tensor([1.5], dtype=torch.bfloat16)})
    renamed = StateModule({"other_scale": torch.tensor(1.5, dtype=torch.bfloat16)})

    scalar_hash = runner._module_tensor_sha256(scalar, torch)

    assert len(scalar_hash) == 64
    assert scalar_hash == runner._module_tensor_sha256(scalar, torch)
    assert scalar_hash != runner._module_tensor_sha256(length_one, torch)
    assert scalar_hash != runner._module_tensor_sha256(renamed, torch)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor([1.5, -2.0], dtype=torch.bfloat16),
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        torch.tensor([[1.25, -0.5], [0.0, 8.0]], dtype=torch.float32),
    ],
)
def test_module_hash_vector_and_matrix_bytes_match_prior_serialization(tensor: Any) -> None:
    runner = _load_runner()
    module = StateModule({"weight": tensor})

    assert runner._module_tensor_sha256(module, torch) == _legacy_nonscalar_hash(module)
