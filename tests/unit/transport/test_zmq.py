from types import SimpleNamespace

import msgspec

from prime_rl.transport.types import TrainingBatch, TrainingSample
from prime_rl.transport.zmq import ZMQTrainingBatchReceiver


class _FakeMultiRunManager:
    def __init__(self, run_ids=None):
        run_ids = run_ids or ["run_default"]
        self.used_idxs = list(range(len(run_ids)))
        self.ready_to_update = [False] * len(run_ids)
        self.idx_2_id = dict(enumerate(run_ids))


def _sample(token_id):
    return TrainingSample(
        token_ids=[token_id],
        mask=[True],
        logprobs=[-0.5],
        temperatures=[1.0],
        env_name="env",
    )


def _receiver_with_pending(pending):
    receiver = ZMQTrainingBatchReceiver.__new__(ZMQTrainingBatchReceiver)
    receiver.multi_run_manager = _FakeMultiRunManager()
    receiver._pending = {b"run_default": pending}
    receiver._last_logged_time = 0.0
    receiver._last_logged_ids = None
    receiver._waiting_since = None
    receiver.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    receiver._socket_has_message = lambda: False
    return receiver


def _receiver_with_pending_runs(pending_by_run):
    receiver = ZMQTrainingBatchReceiver.__new__(ZMQTrainingBatchReceiver)
    run_ids = list(pending_by_run)
    receiver.multi_run_manager = _FakeMultiRunManager(run_ids=run_ids)
    receiver._pending = {run_id.encode("utf-8"): pending for run_id, pending in pending_by_run.items()}
    receiver._last_logged_time = 0.0
    receiver._last_logged_ids = None
    receiver._waiting_since = None
    receiver.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    receiver._socket_has_message = lambda: False
    return receiver


def test_zmq_receiver_can_receive_buffered_pending_without_new_socket_event():
    receiver = _receiver_with_pending({(0, False): TrainingBatch(examples=[_sample(1)], step=0)})

    assert receiver.can_receive()
    batches = receiver.receive()

    assert [batch.examples[0].token_ids for batch in batches] == [[1]]


def test_zmq_receiver_orders_preflight_before_final_for_same_step():
    receiver = _receiver_with_pending(
        {
            (0, False): TrainingBatch(examples=[_sample(2)], step=0),
            (0, True): TrainingBatch(examples=[_sample(1)], step=0, preflight_only=True),
        }
    )

    preflight_batches = receiver.receive()
    final_batches = receiver.receive()

    assert [batch.preflight_only for batch in preflight_batches] == [True]
    assert [batch.examples[0].token_ids for batch in preflight_batches] == [[1]]
    assert [batch.preflight_only for batch in final_batches] == [False]
    assert [batch.examples[0].token_ids for batch in final_batches] == [[2]]


def test_zmq_receiver_keeps_preflight_and_final_modes_separate_across_runs():
    receiver = _receiver_with_pending_runs(
        {
            "run-a": {(0, False): TrainingBatch(examples=[_sample(2)], step=0)},
            "run-b": {(0, True): TrainingBatch(examples=[_sample(1)], step=0, preflight_only=True)},
        }
    )

    preflight_batches = receiver.receive()
    final_batches = receiver.receive()

    assert [batch.preflight_only for batch in preflight_batches] == [True]
    assert [batch.run_idx for batch in preflight_batches] == [1]
    assert [batch.examples[0].token_ids for batch in preflight_batches] == [[1]]
    assert [batch.preflight_only for batch in final_batches] == [False]
    assert [batch.run_idx for batch in final_batches] == [0]
    assert [batch.examples[0].token_ids for batch in final_batches] == [[2]]


def test_training_sample_sample_id_round_trips_without_breaking_legacy_payloads():
    sample = _sample(1)
    sample.sample_id = "sample-a"

    encoded = msgspec.msgpack.encode(sample)
    decoded = msgspec.msgpack.decode(encoded, type=TrainingSample)

    assert decoded.sample_id == "sample-a"

    legacy_payload = msgspec.msgpack.encode(
        [sample.token_ids, sample.mask, sample.logprobs, sample.temperatures, sample.env_name]
    )
    legacy_decoded = msgspec.msgpack.decode(legacy_payload, type=TrainingSample)

    assert legacy_decoded.sample_id is None
    assert legacy_decoded.token_ids == [1]
