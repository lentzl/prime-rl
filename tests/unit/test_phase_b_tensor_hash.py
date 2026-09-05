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


def test_suffix_labels_preserve_torch_causal_cross_entropy_and_gradient_path() -> None:
    runner = _load_runner()
    labels = torch.tensor([[-100, -100, 2, 1]], dtype=torch.long)
    logits = torch.tensor(
        [[[0.1, 0.2, 0.3], [0.8, -0.1, 0.4], [-0.2, 0.7, 0.1], [0.3, 0.0, -0.4]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    suffix_labels, keep, span = runner._suffix_loss_arguments(labels)
    suffix_logits = logits[:, -keep:, :]

    full_shift = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
    suffix_shift = torch.nn.functional.pad(suffix_labels, (0, 1), value=-100)[:, 1:]
    full_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), full_shift.reshape(-1), ignore_index=-100)
    suffix_loss = torch.nn.functional.cross_entropy(
        suffix_logits.reshape(-1, 3), suffix_shift.reshape(-1), ignore_index=-100
    )

    assert span == {"first_supervised_label_index": 2, "logit_suffix_start": 1, "logits_to_keep": 3}
    assert torch.equal(full_loss, suffix_loss)
    suffix_loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, -keep:, :]) > 0


def test_gradient_summary_contains_only_detached_python_scalars() -> None:
    runner = _load_runner()
    gradient = torch.tensor([1.0, -2.0], requires_grad=True)

    summary = runner._gradient_summary(gradient, torch)

    assert summary["present"] is True
    assert summary["finite"] is True
    assert summary["nonzero"] is True
    assert type(summary["l2"]) is float
    assert all(not torch.is_tensor(value) for value in summary.values())
