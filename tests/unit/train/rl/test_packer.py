from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

import prime_rl.trainer.runs as runs
from prime_rl.configs.shared import FileSystemTransportConfig
from prime_rl.trainer.rl.packer import MultiPacker, SinglePacker
from prime_rl.trainer.utils import build_bin_cost
from prime_rl.transport.types import TrainingBatch, TrainingSample


class _FakeMultiRunManager:
    def __init__(self, output_dir: Path, max_runs: int = 1):
        self.output_dir = output_dir
        self.max_runs = max_runs
        self.idx_2_id = {0: "run_test123"}
        self.id_2_idx = {"run_test123": 0}
        self.used_idxs = [0]
        self.ready_to_update = [False] * max_runs
        self.progress = {idx: _progress() for idx in range(max_runs)}
        self.forgotten_hooks = []

    def discover_runs(self):
        pass

    def register_forgotten_hook(self, hook):
        self.forgotten_hooks.append(hook)


def _progress():
    class Progress:
        step = 0
        total_tokens = 0
        total_samples = 0

    return Progress()


def make_training_sample(sample_id: str | None = None) -> TrainingSample:
    return TrainingSample(
        token_ids=[1, 2],
        mask=[False, True],
        logprobs=[0.0, -0.1],
        temperatures=[1.0, 1.0],
        advantages=[0.0, 1.0],
        env_name="test-env",
        sample_id=sample_id,
    )


def make_sample_with_position_ids(position_ids) -> SimpleNamespace:
    sample = make_training_sample()
    return SimpleNamespace(
        token_ids=sample.token_ids,
        mask=sample.mask,
        logprobs=sample.logprobs,
        temperatures=sample.temperatures,
        advantages=sample.advantages,
        env_name=sample.env_name,
        sample_id=sample.sample_id,
        rl_weights=sample.rl_weights,
        ce_weights=sample.ce_weights,
        ref_kl_weights=sample.ref_kl_weights,
        ref_logprobs=sample.ref_logprobs,
        sdpo_topk_token_ids=sample.sdpo_topk_token_ids,
        sdpo_topk_logprobs=sample.sdpo_topk_logprobs,
        sdpo_weights=sample.sdpo_weights,
        sdpo_rollout_is_weights=sample.sdpo_rollout_is_weights,
        position_ids=position_ids,
    )


def _validate(sample: TrainingSample, *, preflight_only: bool = False) -> tuple[bool, str | None]:
    packer = object.__new__(MultiPacker)
    packer.seq_len = 16
    return packer._validate_sample(sample, preflight_only=preflight_only)


def _validate_batch(samples: list[TrainingSample], *, preflight_only: bool = False) -> tuple[bool, str | None]:
    packer = object.__new__(MultiPacker)
    return packer._validate_batch_sample_identity(samples, preflight_only=preflight_only)


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        (1, "non-list token_ids"),
        ([1, True], "non-integer token_ids"),
        ([1, "2"], "non-integer token_ids"),
        ([1, -2], "negative token_ids"),
    ],
)
def test_packer_validate_sample_rejects_malformed_token_ids(token_ids, message: str) -> None:
    sample = make_training_sample()
    sample.token_ids = token_ids

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert message in reason


@pytest.mark.parametrize(
    "field_name",
    ["rl_weights", "ce_weights", "ref_kl_weights", "sdpo_weights", "sdpo_rollout_is_weights"],
)
def test_packer_validate_sample_rejects_misaligned_weight_streams(field_name: str) -> None:
    sample = make_training_sample()
    setattr(sample, field_name, [1.0])

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert field_name in reason
    assert "token_ids length" in reason


def test_packer_validate_sample_rejects_misaligned_advantages() -> None:
    sample = make_training_sample()
    sample.advantages = [1.0]

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert "advantages length" in reason
    assert "token_ids length" in reason


@pytest.mark.parametrize(
    ("position_ids", "message"),
    [
        ((0, 1), "non-list position_ids"),
        ([0], "position_ids length != token_ids length"),
        ([0, True], "non-integer position_ids"),
        ([0, "1"], "non-integer position_ids"),
        ([0, -1], "negative position_ids"),
    ],
)
def test_packer_validate_sample_rejects_malformed_position_ids(position_ids, message: str) -> None:
    sample = make_sample_with_position_ids(position_ids)

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert message in reason


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("rl_weights", True),
        ("ce_weights", "1.0"),
        ("ref_kl_weights", float("nan")),
        ("sdpo_weights", True),
        ("sdpo_rollout_is_weights", "1.0"),
    ],
)
def test_packer_validate_sample_rejects_non_numeric_component_streams(field_name: str, bad_value) -> None:
    sample = make_training_sample()
    setattr(sample, field_name, [0.0, bad_value])

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert f"non-finite or non-numeric {field_name}" in reason


def test_packer_validate_sample_rejects_nonzero_sdpo_weight_outside_mask() -> None:
    sample = make_training_sample()
    sample.sdpo_weights = [1.0, 0.0]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "nonzero sdpo_weights outside mask at token 0" in reason


def test_packer_validate_sample_rejects_negative_sdpo_weight() -> None:
    sample = make_training_sample()
    sample.sdpo_weights = [0.0, -0.25]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "negative sdpo_weights at token 1" in reason


def test_packer_validate_sample_rejects_non_list_sdpo_weights_without_crashing() -> None:
    sample = make_training_sample()
    sample.sdpo_weights = 1.0

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "non-list sdpo_weights" in reason


def test_packer_validate_sample_rejects_nonzero_sdpo_rollout_is_weight_outside_sdpo_component() -> None:
    sample = make_training_sample()
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_rollout_is_weights = [0.5, 1.25]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "nonzero sdpo_rollout_is_weights outside SDPO component at token 0" in reason


@pytest.mark.parametrize("bad_sample_id", ["", "   ", 123])
def test_packer_validate_sample_rejects_malformed_sample_id(bad_sample_id) -> None:
    sample = make_training_sample(sample_id=bad_sample_id)

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert "malformed sample_id" in reason


def test_packer_validate_preflight_sdpo_sample_requires_sample_id() -> None:
    sample = make_training_sample()
    sample.sdpo_weights = [0.0, 1.0]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "SDPO sample without sample_id" in reason


def test_packer_validate_final_sdpo_sample_requires_sample_id() -> None:
    sample = make_training_sample()
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_topk_token_ids = [[0, 0], [10, 11]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [-0.25, -1.5]]

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert "SDPO sample without sample_id" in reason


@pytest.mark.parametrize("bad_env_name", ["", "   ", 123])
def test_packer_validate_sdpo_sample_requires_env_name(bad_env_name) -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.env_name = bad_env_name
    sample.sdpo_weights = [0.0, 1.0]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "SDPO sample without env_name" in reason


def test_packer_validate_preflight_sdpo_sample_accepts_sample_id() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]

    valid, reason = _validate(sample, preflight_only=True)

    assert valid
    assert reason is None


def test_packer_validate_preflight_sdpo_sample_rejects_nonzero_teacher_placeholder_ids() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_topk_token_ids = [[0, 0], [10, 11]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [0.0, 0.0]]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "nonzero placeholder sdpo_topk_token_ids row 1" in reason


def test_packer_validate_batch_rejects_duplicate_sdpo_sample_ids_in_same_phase() -> None:
    first = make_training_sample(sample_id="sample-a")
    first.sdpo_weights = [0.0, 1.0]
    second = make_training_sample(sample_id="sample-a")
    second.sdpo_weights = [0.0, 1.0]

    valid, reason = _validate_batch([first, second], preflight_only=True)

    assert not valid
    assert reason is not None
    assert "duplicate sample_id 'sample-a'" in reason
    assert "same run step" in reason


def test_packer_validate_batch_allows_duplicate_non_sdpo_sample_ids() -> None:
    first = make_training_sample(sample_id="sample-a")
    second = make_training_sample(sample_id="sample-a")

    valid, reason = _validate_batch([first, second])

    assert valid
    assert reason is None


def test_packer_validate_batch_allows_sdpo_sample_id_reuse_across_preflight_and_final_phases() -> None:
    preflight = make_training_sample(sample_id="sample-a")
    preflight.sdpo_weights = [0.0, 1.0]
    final = make_training_sample(sample_id="sample-a")
    final.sdpo_weights = [0.0, 1.0]
    final.sdpo_topk_token_ids = [[0, 0], [10, 11]]
    final.sdpo_topk_logprobs = [[0.0, 0.0], [-0.25, -1.5]]

    preflight_valid, preflight_reason = _validate_batch([preflight], preflight_only=True)
    final_valid, final_reason = _validate_batch([final])

    assert preflight_valid
    assert preflight_reason is None
    assert final_valid
    assert final_reason is None


def test_packer_validate_sample_rejects_negative_sdpo_rollout_is_weight() -> None:
    sample = make_training_sample()
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_rollout_is_weights = [0.0, -0.25]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "negative sdpo_rollout_is_weights at token 1" in reason


def test_packer_validate_sample_accepts_sdpo_rollout_is_weight_on_sdpo_component() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_rollout_is_weights = [0.0, 1.25]

    valid, reason = _validate(sample, preflight_only=True)

    assert valid
    assert reason is None


def test_packer_validate_sample_rejects_integer_sdpo_rollout_is_weight() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_rollout_is_weights = [0.0, 1]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "non-floating sdpo_rollout_is_weights at token 1" in reason


def test_packer_validate_sample_rejects_unpaired_sdpo_topk_streams() -> None:
    sample = make_training_sample()
    sample.sdpo_topk_token_ids = [[0, 0], [10, 11]]

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert "sdpo_topk_logprobs" in reason
    assert "paired SDPO top-k" in reason


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("sdpo_topk_token_ids", 1.0, "non-list sdpo_topk_token_ids"),
        ("sdpo_topk_logprobs", 1.0, "non-list sdpo_topk_logprobs"),
    ],
)
def test_packer_validate_sample_rejects_non_list_sdpo_topk_streams(field_name, bad_value, message) -> None:
    sample = make_training_sample()
    sample.sdpo_topk_token_ids = [[0, 0], [10, 11]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [-0.5, -2.0]]
    setattr(sample, field_name, bad_value)

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert message in reason


@pytest.mark.parametrize(
    ("token_rows", "logprob_rows", "message"),
    [
        ([[0, 0]], [[0.0, 0.0], [-0.5, -2.0]], "sdpo_topk_token_ids length"),
        ([[0, 0], [10, 11]], [[0.0, 0.0]], "sdpo_topk_logprobs length"),
        ([0, [10, 11]], [[0.0, 0.0], [-0.5, -2.0]], "non-list sdpo_topk_token_ids row 0"),
        ([[0, 0], (10, 11)], [[0.0, 0.0], [-0.5, -2.0]], "non-list sdpo_topk_token_ids row 1"),
        ([[0, 0], [10, 11]], [[0.0, 0.0], (-0.5, -2.0)], "non-list sdpo_topk_logprobs row 1"),
        ([[0, 0], [10]], [[0.0, 0.0], [-0.5, -2.0]], "ragged sdpo_topk_token_ids"),
        ([[0, 0], [10, 11]], [[0.0, 0.0], [-0.1]], "sdpo_topk_logprobs row"),
        ([[], []], [[], []], "empty SDPO top-k rows"),
    ],
)
def test_packer_validate_sample_rejects_malformed_sdpo_topk_streams(
    token_rows: list[list[int]], logprob_rows: list[list[float]], message: str
) -> None:
    sample = make_training_sample()
    sample.sdpo_topk_token_ids = token_rows
    sample.sdpo_topk_logprobs = logprob_rows

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert message in reason


def test_packer_validate_final_sdpo_sample_requires_transport_topk_support() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert "final SDPO sample" in reason
    assert "transported SDPO top-k support" in reason


def test_packer_validate_preflight_sdpo_sample_allows_missing_transport_topk_support() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]

    valid, reason = _validate(sample, preflight_only=True)

    assert valid
    assert reason is None


def test_packer_validate_preflight_sdpo_sample_rejects_transported_topk_support() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_topk_token_ids = [[0, 0], [10, 11]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [-0.5, -2.0]]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "preflight SDPO sample with transported top-k support at weighted token 1" in reason


def test_packer_validate_final_sdpo_sample_accepts_transport_topk_support() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_topk_token_ids = [[0, 0], [10, 11]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [-0.5, -2.0]]

    valid, reason = _validate(sample)

    assert valid
    assert reason is None


def test_packer_validate_sdpo_sample_rejects_unweighted_non_placeholder_support() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_topk_token_ids = [[20, 21], [10, 11]]
    sample.sdpo_topk_logprobs = [[-0.7, -1.7], [-0.5, -2.0]]

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert "non-placeholder SDPO top-k support at unweighted token 0" in reason


def test_packer_validate_sdpo_sample_accepts_unweighted_placeholder_support() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_topk_token_ids = [[0, 0], [10, 11]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [-0.5, -2.0]]

    valid, reason = _validate(sample)

    assert valid
    assert reason is None


def test_packer_validate_sample_rejects_non_integer_sdpo_topk_ids() -> None:
    sample = make_training_sample()
    sample.sdpo_topk_token_ids = [[0, 0], [10, True]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [-0.5, -2.0]]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "non-integer sdpo_topk_token_ids row 1" in reason


def test_packer_validate_sample_rejects_negative_sdpo_topk_ids() -> None:
    sample = make_training_sample()
    sample.sdpo_topk_token_ids = [[0, 0], [10, -11]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [-0.5, -2.0]]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "negative sdpo_topk_token_ids row 1" in reason


def test_packer_validate_sample_rejects_nonfinite_sdpo_topk_logprobs() -> None:
    sample = make_training_sample()
    sample.sdpo_topk_token_ids = [[0, 0], [10, 11]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [float("nan"), -2.0]]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "non-finite or non-numeric sdpo_topk_logprobs row 1" in reason


def test_packer_validate_sample_rejects_boolean_sdpo_topk_logprobs() -> None:
    sample = make_training_sample()
    sample.sdpo_topk_token_ids = [[0, 0], [10, 11]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [False, 0.0]]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "non-finite or non-numeric sdpo_topk_logprobs row 1" in reason


def test_packer_validate_sample_rejects_integer_sdpo_topk_logprobs() -> None:
    sample = make_training_sample()
    sample.sdpo_topk_token_ids = [[0, 0], [10, 11]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [-1, -2]]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "non-floating sdpo_topk_logprobs row 1" in reason


@pytest.mark.parametrize("sdpo_weights", [None, [0.0, 0.0]])
def test_packer_validate_sample_rejects_sdpo_topk_streams_without_active_component(sdpo_weights) -> None:
    sample = make_training_sample()
    sample.sdpo_weights = sdpo_weights
    sample.sdpo_topk_token_ids = [[0, 0], [0, 0]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [0.0, 0.0]]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "SDPO top-k streams without nonzero sdpo_weights" in reason


def test_packer_validate_sample_allows_duplicate_placeholder_sdpo_topk_ids() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_topk_token_ids = [[0, 0], [0, 0]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [0.0, 0.0]]

    valid, reason = _validate(sample, preflight_only=True)

    assert valid
    assert reason is None


def test_packer_validate_sample_rejects_nonzero_placeholder_sdpo_topk_ids() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_topk_token_ids = [[0, 0], [10, 11]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [0.0, 0.0]]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "nonzero placeholder sdpo_topk_token_ids row 1" in reason


def test_packer_validate_sample_accepts_token_id_zero_with_real_logprobs() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_topk_token_ids = [[0], [0]]
    sample.sdpo_topk_logprobs = [[0.0], [-0.5]]

    valid, reason = _validate(sample, preflight_only=False)

    assert valid
    assert reason is None


def test_packer_validate_sample_rejects_duplicate_supported_sdpo_topk_ids() -> None:
    sample = make_training_sample()
    sample.sdpo_topk_token_ids = [[0, 0], [10, 10]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [-0.5, -2.0]]

    valid, reason = _validate(sample, preflight_only=True)

    assert not valid
    assert reason is not None
    assert "duplicate sdpo_topk_token_ids row 1" in reason


def test_packer_validate_final_sdpo_sample_rejects_weighted_logprob_mass_above_one() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_topk_token_ids = [[0, 0], [10, 11]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [-0.1, -2.0]]

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert "top-k logprob probability mass > 1 at weighted token 1" in reason


def test_packer_validate_final_sdpo_sample_rejects_weighted_placeholder_support() -> None:
    sample = make_training_sample(sample_id="sample-a")
    sample.sdpo_weights = [0.0, 1.0]
    sample.sdpo_topk_token_ids = [[0, 0], [0, 0]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [0.0, 0.0]]

    valid, reason = _validate(sample)

    assert not valid
    assert reason is not None
    assert "placeholder top-k logprobs at weighted token 1" in reason


def test_packer_progress_updates_once_per_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeMultiRunManager(tmp_path)
    runs._MULTI_RUN_MANAGER = manager
    run_idx = 0

    class DummyReceiver:
        def receive(self):
            return []

        def reset_run(self, idx: int) -> None:
            pass

    class DummySender:
        def __init__(self):
            self.sent = []

        def send(self, micro_batch_grid):
            self.sent.append(micro_batch_grid)

    sender_holder: dict[str, DummySender] = {}

    def fake_receiver(_config):
        return DummyReceiver()

    def fake_sender(_output_dir, _data_world_size, _current_step, _config):
        sender = DummySender()
        sender_holder["sender"] = sender
        return sender

    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_training_batch_receiver", fake_receiver)
    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_micro_batch_sender", fake_sender)

    packer = MultiPacker(
        dp_world_size=1,
        seq_len=4,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        bin_cost=build_bin_cost(None),
        start_step=0,
    )

    packer.buffers[run_idx].append((make_training_sample(), 0, False))
    packer.buffers[run_idx].append((make_training_sample(), 0, False))

    packer.pack()

    progress = manager.progress[run_idx]
    assert progress.total_samples == 2
    assert progress.total_tokens == 4
    assert progress.step == 1

    sender = sender_holder["sender"]
    assert len(sender.sent) == 1
    assert len(sender.sent[0][0]) == 1
    micro_batch = sender.sent[0][0][0]
    assert micro_batch.run_id == "run_test123"
    assert micro_batch.run_step == 0
    assert micro_batch.preflight_only is False


def test_single_packer_sdpo_preflight_preserves_metadata_without_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _FakeMultiRunManager(tmp_path)
    runs._MULTI_RUN_MANAGER = manager
    sample = make_training_sample("sample-a")
    sample.sdpo_weights = [0.0, 1.0]

    class DummyReceiver:
        def receive(self):
            return [TrainingBatch(examples=[sample], step=0, run_idx=0, preflight_only=True)]

    class DummySender:
        def __init__(self):
            self.sent = []

        def send(self, micro_batch_grid):
            self.sent.append(micro_batch_grid)

    sender_holder: dict[str, DummySender] = {}

    def fake_receiver(_config):
        return DummyReceiver()

    def fake_sender(_output_dir, _data_world_size, _current_step, _config):
        sender = DummySender()
        sender_holder["sender"] = sender
        return sender

    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_training_batch_receiver", fake_receiver)
    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_micro_batch_sender", fake_sender)

    packer = SinglePacker(
        dp_world_size=1,
        seq_len=4,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        bin_cost=build_bin_cost(None),
        start_step=0,
    )

    packer.pack()

    progress = manager.progress[0]
    assert progress.step == 0
    assert manager.ready_to_update[0] is False

    sender = sender_holder["sender"]
    assert len(sender.sent) == 1
    micro_batch = sender.sent[0][0][0]
    assert micro_batch.run_id == "run_test123"
    assert micro_batch.run_step == 0
    assert micro_batch.preflight_only is True
    assert micro_batch.preflight_step_complete is True
    assert micro_batch.sample_ids == ["sample-a"]
    assert micro_batch.sdpo_weights == [0.0, 1.0]
    assert micro_batch.sdpo_topk_token_ids is None


def test_packer_preflight_does_not_advance_run_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeMultiRunManager(tmp_path)
    runs._MULTI_RUN_MANAGER = manager
    run_idx = 0

    class DummyReceiver:
        def receive(self):
            return []

        def reset_run(self, idx: int) -> None:
            pass

    class DummySender:
        def __init__(self):
            self.sent = []

        def send(self, micro_batch_grid):
            self.sent.append(micro_batch_grid)

    sender_holder: dict[str, DummySender] = {}

    def fake_receiver(_config):
        return DummyReceiver()

    def fake_sender(_output_dir, _data_world_size, _current_step, _config):
        sender = DummySender()
        sender_holder["sender"] = sender
        return sender

    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_training_batch_receiver", fake_receiver)
    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_micro_batch_sender", fake_sender)

    packer = MultiPacker(
        dp_world_size=1,
        seq_len=4,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        bin_cost=build_bin_cost(None),
        start_step=0,
    )

    packer.buffers[run_idx].append((make_training_sample("sample-a"), 0, True))
    packer.buffers[run_idx].append((make_training_sample("sample-b"), 0, True))

    packer.pack()

    progress = manager.progress[run_idx]
    assert progress.total_samples == 0
    assert progress.total_tokens == 0
    assert progress.step == 0
    assert manager.ready_to_update[run_idx] is False

    sender = sender_holder["sender"]
    assert len(sender.sent) == 1
    micro_batch = sender.sent[0][0][0]
    assert micro_batch.run_id == "run_test123"
    assert micro_batch.run_step == 0
    assert micro_batch.preflight_only is True
    assert micro_batch.preflight_step_complete is True
    assert micro_batch.sample_ids == ["sample-a", "sample-b"]


def test_multi_packer_marks_split_preflight_complete_only_after_buffer_drains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _FakeMultiRunManager(tmp_path)
    runs._MULTI_RUN_MANAGER = manager
    run_idx = 0

    class DummyReceiver:
        def receive(self):
            return []

        def reset_run(self, idx: int) -> None:
            pass

    class DummySender:
        def __init__(self):
            self.sent = []

        def send(self, micro_batch_grid):
            self.sent.append(micro_batch_grid)

    sender_holder: dict[str, DummySender] = {}

    def fake_receiver(_config):
        return DummyReceiver()

    def fake_sender(_output_dir, _data_world_size, _current_step, _config):
        sender = DummySender()
        sender_holder["sender"] = sender
        return sender

    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_training_batch_receiver", fake_receiver)
    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_micro_batch_sender", fake_sender)

    packer = MultiPacker(
        dp_world_size=1,
        seq_len=4,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        bin_cost=build_bin_cost(None),
        start_step=0,
    )

    packer.buffers[run_idx].append((make_training_sample("sample-a"), 0, True))
    packer.buffers[run_idx].append((make_training_sample("sample-b"), 0, True))
    packer.buffers[run_idx].append((make_training_sample("sample-c"), 0, True))

    packer.pack()
    first_micro_batch = sender_holder["sender"].sent[0][0][0]

    assert first_micro_batch.sample_ids == ["sample-a", "sample-b"]
    assert first_micro_batch.preflight_only is True
    assert first_micro_batch.preflight_step_complete is False
    assert manager.progress[run_idx].step == 0
    assert manager.ready_to_update[run_idx] is False

    packer.pack()
    second_micro_batch = sender_holder["sender"].sent[1][0][0]

    assert second_micro_batch.sample_ids == ["sample-c"]
    assert second_micro_batch.preflight_only is True
    assert second_micro_batch.preflight_step_complete is True
    assert manager.progress[run_idx].step == 0
    assert manager.ready_to_update[run_idx] is False


def test_multi_packer_readiness_counts_preflight_and_final_modes_separately(tmp_path: Path) -> None:
    manager = _FakeMultiRunManager(tmp_path, max_runs=2)
    manager.idx_2_id[1] = "run_test456"
    manager.id_2_idx["run_test456"] = 1
    manager.used_idxs = [0, 1]
    packer = object.__new__(MultiPacker)
    packer.multi_run_manager = manager
    packer.seq_len = 4
    packer.dp_world_size = 1
    packer.buffers = [
        deque([(make_training_sample("sample-a"), 0, True)]),
        deque([(make_training_sample("sample-b"), 0, False)]),
    ]

    assert packer._count_tokens() == 4
    assert packer._count_tokens(preflight_only=True) == 2
    assert packer._count_tokens(preflight_only=False) == 2
    assert not packer._has_enough_tokens()

    packer.buffers[0].append((make_training_sample("sample-c"), 0, True))

    assert packer._count_tokens(preflight_only=True) == 4
    assert packer._has_enough_tokens()


def test_multi_packer_readiness_prefers_pending_preflight_over_full_final_batch(tmp_path: Path) -> None:
    manager = _FakeMultiRunManager(tmp_path, max_runs=2)
    manager.idx_2_id[1] = "run_test456"
    manager.id_2_idx["run_test456"] = 1
    manager.used_idxs = [0, 1]
    packer = object.__new__(MultiPacker)
    packer.multi_run_manager = manager
    packer.seq_len = 4
    packer.dp_world_size = 1
    packer.buffers = [
        deque([(make_training_sample("sample-a"), 0, False), (make_training_sample("sample-b"), 0, False)]),
        deque([(make_training_sample("sample-c"), 0, True)]),
    ]

    assert packer._count_tokens(preflight_only=False) == 4
    assert packer._count_tokens(preflight_only=True) == 2
    assert not packer._has_enough_tokens()

    packer.buffers[1].append((make_training_sample("sample-d"), 0, True))

    assert packer._count_tokens(preflight_only=True) == 4
    assert packer._has_enough_tokens()


def test_multi_packer_selection_prefers_pending_preflight_over_earlier_final_run(tmp_path: Path) -> None:
    manager = _FakeMultiRunManager(tmp_path, max_runs=2)
    manager.idx_2_id[1] = "run_test456"
    manager.id_2_idx["run_test456"] = 1
    manager.used_idxs = [0, 1]
    packer = object.__new__(MultiPacker)
    packer.multi_run_manager = manager
    packer._round_robin_position = 0
    packer.buffers = [
        deque([(make_training_sample("sample-a"), 0, False)]),
        deque([(make_training_sample("sample-b"), 0, True)]),
    ]

    selected = packer._select_samples_round_robin(token_budget=4)

    assert [(run_idx, sample.sample_id, preflight_only) for run_idx, sample, _step, preflight_only in selected] == [
        (1, "sample-b", True)
    ]
    assert [sample.sample_id for sample, _step, _preflight_only in packer.buffers[0]] == ["sample-a"]
    assert packer.buffers[1] == deque()
