import pytest
import torch

from prime_rl.trainer.weights import merge_lora_weights


def test_merge_lora_weights_applies_scaled_update_without_adapter_keys():
    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lora_a = torch.tensor([[1.0, -1.0]])
    lora_b = torch.tensor([[2.0], [3.0]])
    state_dict = {"model.proj.weight": base.clone()}
    adapter = {
        "model.proj.lora_A.weight": lora_a,
        "model.proj.lora_B.weight": lora_b,
    }

    merged = merge_lora_weights(state_dict, adapter, scaling=0.5)

    expected = base + 0.5 * (lora_b @ lora_a)
    torch.testing.assert_close(merged["model.proj.weight"], expected)
    assert set(merged) == {"model.proj.weight"}


def test_merge_lora_weights_preserves_base_dtype():
    state_dict = {"model.proj.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}
    adapter = {
        "model.proj.lora_A.weight": torch.ones(1, 2),
        "model.proj.lora_B.weight": torch.ones(2, 1),
    }

    merged = merge_lora_weights(state_dict, adapter, scaling=1.0)

    assert merged["model.proj.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(merged["model.proj.weight"], torch.ones(2, 2, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    ("adapter", "error"),
    [
        ({"model.proj.lora_A.weight": torch.ones(1, 2)}, ValueError),
        (
            {
                "missing.proj.lora_A.weight": torch.ones(1, 2),
                "missing.proj.lora_B.weight": torch.ones(2, 1),
            },
            KeyError,
        ),
    ],
)
def test_merge_lora_weights_rejects_incomplete_or_unknown_adapters(adapter, error):
    with pytest.raises(error):
        merge_lora_weights({"model.proj.weight": torch.zeros(2, 2)}, adapter, 1.0)
