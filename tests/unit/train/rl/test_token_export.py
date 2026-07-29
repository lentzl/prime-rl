import torch

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.trainer.rl.token_export import _compute_export_tensors


def test_novelty_only_tokens_export_policy_ratio_fields():
    micro_batch = {
        "inference_logprobs": torch.tensor([[-1.1, -1.2]]),
        "loss_mask": torch.tensor([[True, True]]),
        "advantages": torch.zeros(1, 2),
        "rl_weights": torch.zeros(1, 2),
        "ref_kl_weights": None,
        "sdpo_weights": torch.zeros(1, 2),
        "novelty_weights": torch.ones(1, 2),
    }

    fields = _compute_export_tensors(
        micro_batch,
        trainer_logprobs=torch.tensor([[-1.0, -1.0]]),
        loss_config=DefaultLossConfig(),
    )

    assert fields["importance_ratio"] is not None
    assert fields["mismatch_kl"] is not None
