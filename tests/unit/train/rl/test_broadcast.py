from prime_rl.trainer.rl.broadcast import in_memory_broadcast_is_unused


def test_in_memory_broadcast_respects_synchronous_batch_lead():
    assert not in_memory_broadcast_is_unused(progress_step=1, max_steps=2, max_train_batch_lead=0)
    assert in_memory_broadcast_is_unused(progress_step=2, max_steps=2, max_train_batch_lead=0)


def test_in_memory_broadcast_preserves_default_one_batch_lead():
    assert in_memory_broadcast_is_unused(progress_step=1, max_steps=2, max_train_batch_lead=1)
    assert not in_memory_broadcast_is_unused(progress_step=10, max_steps=None, max_train_batch_lead=1)
