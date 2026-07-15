from prime_rl.trainer.rl.broadcast import nccl_broadcast_is_unused


def test_nccl_broadcast_respects_synchronous_batch_lead():
    assert not nccl_broadcast_is_unused(progress_step=1, max_steps=2, max_train_batch_lead=0)
    assert nccl_broadcast_is_unused(progress_step=2, max_steps=2, max_train_batch_lead=0)


def test_nccl_broadcast_preserves_default_one_batch_lead():
    assert nccl_broadcast_is_unused(progress_step=1, max_steps=2, max_train_batch_lead=1)
    assert not nccl_broadcast_is_unused(progress_step=10, max_steps=None, max_train_batch_lead=1)
