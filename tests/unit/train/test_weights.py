import torch
from safetensors.torch import load_file

from prime_rl.trainer.weights import save_state_dict


def test_save_state_dict_handles_shared_safetensor_storage(tmp_path):
    shared = torch.arange(8, dtype=torch.bfloat16)

    save_state_dict(
        {"embed.weight": shared, "lm_head.weight": shared},
        tmp_path,
        save_format="safetensors",
        save_sharded=True,
    )

    saved = load_file(tmp_path / "model.safetensors")
    torch.testing.assert_close(saved["embed.weight"], shared)
    torch.testing.assert_close(saved["lm_head.weight"], shared)
