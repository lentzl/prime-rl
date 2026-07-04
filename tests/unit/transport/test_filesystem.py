import asyncio
from types import SimpleNamespace

from prime_rl.transport.filesystem import FileSystemTrainingBatchReceiver, FileSystemTrainingBatchSender
from prime_rl.transport.types import TrainingBatch, TrainingSample


class _FakeMultiRunManager:
    def __init__(self, output_dir, run_ids=None):
        run_ids = run_ids or ["run_default"]
        self.output_dir = output_dir
        self.used_idxs = list(range(len(run_ids)))
        self.ready_to_update = [False] * len(run_ids)
        self.idx_2_id = dict(enumerate(run_ids))
        self.progress = {idx: SimpleNamespace(step=0) for idx in self.used_idxs}

    def get_run_dir(self, idx):
        return self.output_dir / self.idx_2_id[idx]


def _sample(token_id):
    return TrainingSample(
        token_ids=[token_id],
        mask=[True],
        logprobs=[-0.5],
        temperatures=[1.0],
        env_name="env",
    )


def test_filesystem_receiver_reads_preflight_then_final_batch_on_same_step(tmp_path, monkeypatch):
    manager = _FakeMultiRunManager(tmp_path)
    monkeypatch.setattr("prime_rl.transport.filesystem.get_multi_run_manager", lambda: manager)

    sender = FileSystemTrainingBatchSender(tmp_path / "run_default")
    asyncio.run(sender.send(TrainingBatch(examples=[_sample(1)], step=0, preflight_only=True)))
    asyncio.run(sender.send(TrainingBatch(examples=[_sample(2)], step=0)))

    receiver = FileSystemTrainingBatchReceiver()

    preflight_batches = receiver.receive()

    assert [batch.preflight_only for batch in preflight_batches] == [True]
    assert [batch.examples[0].token_ids for batch in preflight_batches] == [[1]]
    assert receiver._get_received_step(0) == 0

    final_batches = receiver.receive()

    assert [batch.preflight_only for batch in final_batches] == [False]
    assert [batch.examples[0].token_ids for batch in final_batches] == [[2]]
    assert receiver._get_received_step(0) == 1


def test_filesystem_receiver_does_not_replay_preflight_after_final_batch(tmp_path, monkeypatch):
    manager = _FakeMultiRunManager(tmp_path)
    monkeypatch.setattr("prime_rl.transport.filesystem.get_multi_run_manager", lambda: manager)

    sender = FileSystemTrainingBatchSender(tmp_path / "run_default")
    asyncio.run(sender.send(TrainingBatch(examples=[_sample(1)], step=0, preflight_only=True)))
    asyncio.run(sender.send(TrainingBatch(examples=[_sample(2)], step=0)))

    receiver = FileSystemTrainingBatchReceiver()
    receiver.receive()
    receiver.receive()

    assert receiver.receive() == []


def test_filesystem_receiver_keeps_preflight_and_final_modes_separate_across_runs(tmp_path, monkeypatch):
    manager = _FakeMultiRunManager(tmp_path, run_ids=["run-a", "run-b"])
    monkeypatch.setattr("prime_rl.transport.filesystem.get_multi_run_manager", lambda: manager)

    sender_a = FileSystemTrainingBatchSender(tmp_path / "run-a")
    sender_b = FileSystemTrainingBatchSender(tmp_path / "run-b")
    asyncio.run(sender_a.send(TrainingBatch(examples=[_sample(2)], step=0)))
    asyncio.run(sender_b.send(TrainingBatch(examples=[_sample(1)], step=0, preflight_only=True)))

    receiver = FileSystemTrainingBatchReceiver()

    preflight_batches = receiver.receive()
    final_batches = receiver.receive()

    assert [batch.preflight_only for batch in preflight_batches] == [True]
    assert [batch.run_idx for batch in preflight_batches] == [1]
    assert [batch.examples[0].token_ids for batch in preflight_batches] == [[1]]
    assert [batch.preflight_only for batch in final_batches] == [False]
    assert [batch.run_idx for batch in final_batches] == [0]
    assert [batch.examples[0].token_ids for batch in final_batches] == [[2]]
