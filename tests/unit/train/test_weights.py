import pytest
import torch

from prime_rl.trainer.weights import load_state_dict, save_state_dict


@pytest.mark.parametrize("save_sharded", [False, True])
def test_save_state_dict_clones_shared_storage_for_safetensors(tmp_path, save_sharded):
    tied_weight = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    state_dict = {
        "model.embed_tokens.weight": tied_weight,
        "lm_head.weight": tied_weight,
    }

    save_state_dict(state_dict, tmp_path, save_format="safetensors", save_sharded=save_sharded)
    restored = load_state_dict(tmp_path)

    assert restored.keys() == state_dict.keys()
    torch.testing.assert_close(restored["model.embed_tokens.weight"], tied_weight)
    torch.testing.assert_close(restored["lm_head.weight"], tied_weight)
