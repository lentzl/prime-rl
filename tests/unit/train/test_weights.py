import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from prime_rl.configs.trainer import LoRAConfig, WeightCheckpointConfig
from prime_rl.trainer.ckpt import WeightCheckpointManager
from prime_rl.trainer.runs import MultiRunManager
from prime_rl.trainer.weights import merge_lora_weights, normalize_peft_lora_state_dict


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


def test_normalize_peft_lora_state_dict_rejects_non_adapter_tensor():
    with pytest.raises(ValueError, match="Unexpected tensor"):
        normalize_peft_lora_state_dict({"base_model.model.model.proj.weight": torch.zeros(2, 2)})


def test_load_run_adapter_validates_metadata_and_copies_parameters(tmp_path):
    adapter_a = torch.tensor([[1.0, 2.0]])
    adapter_b = torch.tensor([[3.0], [4.0]])
    save_file(
        {
            "base_model.model.model.proj.lora_A.weight": adapter_a,
            "base_model.model.model.proj.lora_B.weight": adapter_b,
        },
        tmp_path / "adapter_model.safetensors",
    )
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 1, "lora_alpha": 2.0})
    )

    target_a = torch.nn.Parameter(torch.zeros_like(adapter_a))
    target_b = torch.nn.Parameter(torch.zeros_like(adapter_b))
    manager = object.__new__(MultiRunManager)
    manager.logger = MagicMock()
    manager.get_named_parameters_for_run = MagicMock(
        return_value=[
            ("model.proj.lora_A.weight", target_a),
            ("model.proj.lora_B.weight", target_b),
        ]
    )

    manager.load_run_adapter(0, tmp_path, LoRAConfig(rank=1, alpha=2))

    torch.testing.assert_close(target_a, adapter_a)
    torch.testing.assert_close(target_b, adapter_b)
    manager.logger.info.assert_called_once()


def test_adapter_only_checkpoint_skips_full_weight_gather(tmp_path):
    world = SimpleNamespace(is_master=True)
    config = WeightCheckpointConfig(save_adapter_only=True)
    lora_config = LoRAConfig(rank=16, alpha=32)
    adapter_state = {"base_model.model.proj.lora_A.weight": torch.ones(1, 2)}

    with (
        patch("prime_rl.trainer.ckpt.get_world", return_value=world),
        patch("prime_rl.trainer.ckpt.has_lora_layers", return_value=True),
        patch("prime_rl.trainer.ckpt.torch.distributed.barrier"),
        patch("prime_rl.trainer.ckpt.gather_weights_on_master") as gather_weights,
        patch("prime_rl.trainer.ckpt.save_state_dict") as save_state_dict,
        patch("prime_rl.trainer.ckpt.save_lora_config") as save_lora_config,
    ):
        manager = WeightCheckpointManager(tmp_path, config, lora_config=lora_config)
        manager.get_run_adapter_state_dict = MagicMock(return_value=adapter_state)
        model = MagicMock()

        manager.save(3, model, MagicMock())

    step_path = tmp_path / "weights" / "step_3"
    gather_weights.assert_not_called()
    save_state_dict.assert_called_once_with(
        adapter_state,
        step_path,
        "safetensors",
        save_sharded=False,
        adapter=True,
    )
    save_lora_config.assert_called_once_with(model, step_path, rank=16, alpha=32.0, dropout=0.0)
    assert (step_path / "STABLE").exists()
