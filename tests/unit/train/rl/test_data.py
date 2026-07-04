from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.rl.data import DataLoader
from prime_rl.transport.types import MicroBatch


def test_micro_batch_to_tensor_preserves_sdpo_streams():
    loader = object.__new__(DataLoader)
    loader.multi_run_manager = SimpleNamespace(max_runs=1)

    micro_batch = MicroBatch(
        input_ids=[10, 11, 12],
        loss_mask=[False, True, True],
        advantages=[0.0, 1.0, 1.0],
        inference_logprobs=[0.0, -0.2, -0.3],
        position_ids=[0, 1, 2],
        sequence_lengths=[3],
        sample_ids=["sample-a"],
        temperatures=[1.0, 0.8, 0.8],
        env_names=["sdpo-env", "sdpo-env", "sdpo-env"],
        sdpo_topk_token_ids=[[0, 0], [101, 102], [201, 202]],
        sdpo_topk_logprobs=[[0.0, 0.0], [-0.1, -2.4], [-0.2, -2.1]],
        sdpo_rollout_is_weights=[1.0, 0.75, 1.25],
        sdpo_weights=[0.0, 1.0, 1.0],
        lora_num_tokens=[3],
        run_id="run-sdpo",
        run_step=7,
        preflight_only=True,
        preflight_step_complete=False,
    )

    tensor_batch = loader._micro_batch_to_tensor(micro_batch)

    assert tensor_batch["sdpo_topk_token_ids"].shape == (1, 3, 2)
    assert tensor_batch["sdpo_topk_token_ids"].dtype == torch.long
    assert tensor_batch["sdpo_topk_token_ids"].tolist() == [[[0, 0], [101, 102], [201, 202]]]
    assert tensor_batch["sdpo_topk_logprobs"].shape == (1, 3, 2)
    torch.testing.assert_close(
        tensor_batch["sdpo_topk_logprobs"],
        torch.tensor([[[0.0, 0.0], [-0.1, -2.4], [-0.2, -2.1]]]),
    )
    torch.testing.assert_close(tensor_batch["sdpo_rollout_is_weights"], torch.tensor([[1.0, 0.75, 1.25]]))
    torch.testing.assert_close(tensor_batch["sdpo_weights"], torch.tensor([[0.0, 1.0, 1.0]]))
    assert tensor_batch["sample_ids"] == ["sample-a"]
    assert tensor_batch["run_id"] == "run-sdpo"
    assert tensor_batch["run_step"] == 7
    assert tensor_batch["preflight_only"] is True
    assert tensor_batch["preflight_step_complete"] is False


def test_micro_batch_to_tensor_rejects_integer_sdpo_topk_logprobs():
    loader = object.__new__(DataLoader)
    loader.multi_run_manager = SimpleNamespace(max_runs=1)

    micro_batch = MicroBatch(
        input_ids=[10, 11, 12],
        loss_mask=[False, True, True],
        advantages=[0.0, 1.0, 1.0],
        inference_logprobs=[0.0, -0.2, -0.3],
        position_ids=[0, 1, 2],
        sequence_lengths=[3],
        sample_ids=["sample-a"],
        temperatures=[1.0, 0.8, 0.8],
        env_names=["sdpo-env", "sdpo-env", "sdpo-env"],
        sdpo_topk_token_ids=[[0, 0], [101, 102], [201, 202]],
        sdpo_topk_logprobs=[[0.0, 0.0], [-1, -2], [-0.2, -2.1]],
        sdpo_weights=[0.0, 1.0, 1.0],
        lora_num_tokens=[3],
    )

    with pytest.raises(ValueError, match="sdpo_topk_logprobs row 1 must contain finite floating-point values"):
        loader._micro_batch_to_tensor(micro_batch)


def test_micro_batch_to_tensor_rejects_integer_sdpo_rollout_is_weights():
    loader = object.__new__(DataLoader)
    loader.multi_run_manager = SimpleNamespace(max_runs=1)

    micro_batch = MicroBatch(
        input_ids=[10, 11, 12],
        loss_mask=[False, True, True],
        advantages=[0.0, 1.0, 1.0],
        inference_logprobs=[0.0, -0.2, -0.3],
        position_ids=[0, 1, 2],
        sequence_lengths=[3],
        sample_ids=["sample-a"],
        temperatures=[1.0, 0.8, 0.8],
        env_names=["sdpo-env", "sdpo-env", "sdpo-env"],
        sdpo_rollout_is_weights=[0.0, 1, 1.25],
        sdpo_weights=[0.0, 1.0, 1.0],
        lora_num_tokens=[3],
    )

    with pytest.raises(
        ValueError,
        match=r"sdpo_rollout_is_weights\[1\] must be a floating-point value before tensorization",
    ):
        loader._micro_batch_to_tensor(micro_batch)
