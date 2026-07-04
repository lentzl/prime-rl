import asyncio
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.algo import finalize_batch_preflight
from prime_rl.orchestrator.sdpo_preflight import (
    run_sdpo_student_support_preflight,
    token_export_step_dir,
)
from prime_rl.orchestrator.sdpo_sample_identity import ensure_sdpo_sample_ids


class _FakeTrainEnvs:
    def __init__(self, algorithms):
        self.algorithms = algorithms

    def get(self, env_name):
        return SimpleNamespace(algorithm=self.algorithms[env_name])


class _FakeSender:
    def __init__(self):
        self.sent = []

    async def send(self, batch):
        self.sent.append(batch)


class _FakeSharedPreflightAlgorithm:
    batch_preflight_name = "shared-student-support"

    def __init__(self, *, selected_samples=None, signature=None):
        self.needs_batches = []
        self.select_calls = []
        self.run_calls = []
        self.selected_samples = selected_samples
        self.signature = signature

    def needs_batch_preflight(self, batch):
        self.needs_batches.append(batch)
        return True

    def select_batch_preflight_samples(self, batch, *, samples):
        self.select_calls.append({"env_names": [rollout.env_name for rollout in batch], "samples": samples})
        return self.selected_samples if self.selected_samples is not None else samples

    async def run_batch_preflight(self, batch, *, samples, output_dir, sender, step):
        self.run_calls.append(
            {
                "env_names": [rollout.env_name for rollout in batch],
                "samples": samples,
                "output_dir": output_dir,
                "sender": sender,
                "step": step,
            }
        )

    def batch_preflight_signature(self):
        return self.signature


def _rollout(env_name="sdpo_env", *, filtered=False, samples=None):
    return SimpleNamespace(env_name=env_name, is_filtered=filtered, samples=samples or [])


def test_finalize_batch_preflight_runs_shared_algorithm_hook_once(tmp_path):
    first = _FakeSharedPreflightAlgorithm()
    second = _FakeSharedPreflightAlgorithm()
    train_envs = _FakeTrainEnvs({"env-a": first, "env-b": second})
    sample_a = SimpleNamespace(token_ids=[1])
    sample_b = SimpleNamespace(token_ids=[2])
    sample_filtered = SimpleNamespace(token_ids=[3])
    samples = [sample_a, sample_b, sample_filtered]
    sender = _FakeSender()

    asyncio.run(
        finalize_batch_preflight(
            train_envs,
            [
                _rollout("env-a", samples=[sample_a]),
                _rollout("env-b", samples=[sample_b]),
                _rollout("env-b", filtered=True, samples=[sample_filtered]),
            ],
            samples=samples,
            output_dir=tmp_path,
            sender=sender,
            step=5,
        )
    )

    assert len(first.needs_batches) == 1
    assert len(second.needs_batches) == 1
    assert first.select_calls == [{"env_names": ["env-a"], "samples": [sample_a]}]
    assert second.select_calls == [{"env_names": ["env-b"], "samples": [sample_b]}]
    assert len(first.run_calls) == 1
    assert second.run_calls == []
    assert first.run_calls[0] == {
        "env_names": ["env-a", "env-b"],
        "samples": [sample_a, sample_b],
        "output_dir": tmp_path,
        "sender": sender,
        "step": 5,
    }


def test_finalize_batch_preflight_selects_samples_per_env_before_sharing(tmp_path):
    sample_a = SimpleNamespace(token_ids=[1])
    sample_b = SimpleNamespace(token_ids=[2])
    skipped_a = SimpleNamespace(token_ids=[10])
    skipped_b = SimpleNamespace(token_ids=[20])
    first = _FakeSharedPreflightAlgorithm(selected_samples=[sample_a])
    second = _FakeSharedPreflightAlgorithm(selected_samples=[sample_b])
    train_envs = _FakeTrainEnvs({"env-a": first, "env-b": second})
    sender = _FakeSender()

    asyncio.run(
        finalize_batch_preflight(
            train_envs,
            [_rollout("env-a", samples=[sample_a, skipped_a]), _rollout("env-b", samples=[sample_b, skipped_b])],
            samples=[sample_a, skipped_a, sample_b, skipped_b],
            output_dir=tmp_path,
            sender=sender,
            step=6,
        )
    )

    assert first.select_calls == [{"env_names": ["env-a"], "samples": [sample_a, skipped_a]}]
    assert second.select_calls == [{"env_names": ["env-b"], "samples": [sample_b, skipped_b]}]
    assert len(first.run_calls) == 1
    assert second.run_calls == []
    assert first.run_calls[0]["env_names"] == ["env-a", "env-b"]
    assert first.run_calls[0]["samples"] == [sample_a, sample_b]


def test_finalize_batch_preflight_limits_selection_to_final_training_samples(tmp_path):
    trainer_bound = SimpleNamespace(token_ids=[1])
    not_trainer_bound = SimpleNamespace(token_ids=[2])
    algorithm = _FakeSharedPreflightAlgorithm()
    train_envs = _FakeTrainEnvs({"env-a": algorithm})
    sender = _FakeSender()

    asyncio.run(
        finalize_batch_preflight(
            train_envs,
            [_rollout("env-a", samples=[trainer_bound, not_trainer_bound])],
            samples=[trainer_bound],
            output_dir=tmp_path,
            sender=sender,
            step=6,
        )
    )

    assert algorithm.select_calls == [{"env_names": ["env-a"], "samples": [trainer_bound]}]
    assert len(algorithm.run_calls) == 1
    assert algorithm.run_calls[0]["samples"] == [trainer_bound]


def test_finalize_batch_preflight_skips_hook_when_selector_prunes_all_samples(tmp_path):
    sample = SimpleNamespace(token_ids=[1])
    algorithm = _FakeSharedPreflightAlgorithm(selected_samples=[])
    train_envs = _FakeTrainEnvs({"env-a": algorithm})
    sender = _FakeSender()

    asyncio.run(
        finalize_batch_preflight(
            train_envs,
            [_rollout("env-a", samples=[sample])],
            samples=[sample],
            output_dir=tmp_path,
            sender=sender,
            step=6,
        )
    )

    assert len(algorithm.needs_batches) == 1
    assert algorithm.select_calls == [{"env_names": ["env-a"], "samples": [sample]}]
    assert algorithm.run_calls == []
    assert sender.sent == []


def test_finalize_batch_preflight_rejects_selected_samples_outside_env_scope(tmp_path):
    sample_a = SimpleNamespace(token_ids=[1])
    external = SimpleNamespace(token_ids=[99])
    first = _FakeSharedPreflightAlgorithm(selected_samples=[external])
    train_envs = _FakeTrainEnvs({"env-a": first})
    sender = _FakeSender()

    with pytest.raises(ValueError, match="outside its survivor scope"):
        asyncio.run(
            finalize_batch_preflight(
                train_envs,
                [_rollout("env-a", samples=[sample_a])],
                samples=[sample_a, external],
                output_dir=tmp_path,
                sender=sender,
                step=7,
            )
        )

    assert first.run_calls == []


def test_finalize_batch_preflight_rejects_duplicate_selected_samples(tmp_path):
    sample_a = SimpleNamespace(token_ids=[1])
    first = _FakeSharedPreflightAlgorithm(selected_samples=[sample_a, sample_a])
    train_envs = _FakeTrainEnvs({"env-a": first})
    sender = _FakeSender()

    with pytest.raises(ValueError, match="returned duplicate samples"):
        asyncio.run(
            finalize_batch_preflight(
                train_envs,
                [_rollout("env-a", samples=[sample_a])],
                samples=[sample_a],
                output_dir=tmp_path,
                sender=sender,
                step=7,
            )
        )

    assert first.run_calls == []
    assert sender.sent == []


def test_finalize_batch_preflight_rejects_incompatible_shared_hook_signatures(tmp_path):
    sample_a = SimpleNamespace(token_ids=[1])
    sample_b = SimpleNamespace(token_ids=[2])
    first = _FakeSharedPreflightAlgorithm(signature=("sdpo_student_support", 2))
    second = _FakeSharedPreflightAlgorithm(signature=("sdpo_student_support", 3))
    train_envs = _FakeTrainEnvs({"env-a": first, "env-b": second})
    sender = _FakeSender()

    with pytest.raises(ValueError, match="incompatible algorithm configurations"):
        asyncio.run(
            finalize_batch_preflight(
                train_envs,
                [_rollout("env-a", samples=[sample_a]), _rollout("env-b", samples=[sample_b])],
                samples=[sample_a, sample_b],
                output_dir=tmp_path,
                sender=sender,
                step=8,
            )
        )

    assert first.run_calls == []
    assert second.run_calls == []
    assert sender.sent == []


def test_run_sdpo_student_support_preflight_sends_preflight_batch_and_hydrates(tmp_path, monkeypatch):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    sdpo_sample = SimpleNamespace(
        token_ids=[1, 2],
        mask=[False, True],
        sdpo_weights=[0.0, 1.0],
        env_name="sdpo_env",
        sample_id=None,
    )
    non_sdpo_sample = SimpleNamespace(token_ids=[3, 4], mask=[False, True], sdpo_weights=[0.0, 0.0], sample_id=None)
    samples = [sdpo_sample, non_sdpo_sample]

    export_dir = token_export_step_dir(output_dir, 3)
    export_dir.mkdir(parents=True)
    stale_file = export_dir / "stale.jsonl"
    stale_file.write_text("stale", encoding="utf-8")

    async def fake_wait_for_path(path):
        assert path == export_dir / "STABLE"
        assert not stale_file.exists()
        path.parent.mkdir(parents=True)
        path.touch()

    def fake_load_student_support_records(path, *, require_preflight_only=False):
        assert path == export_dir
        assert require_preflight_only is True
        return ["record"]

    def fake_hydrate_student_support_from_records(
        received_samples,
        records,
        *,
        expected_topk=None,
        require_sample_ids=False,
    ):
        assert received_samples == [sdpo_sample]
        assert records == ["record"]
        assert expected_topk == 2
        assert require_sample_ids is True
        assert received_samples[0].sample_id == "sdpo-preflight-step-3-env-sdpo_env-sample-0"
        received_samples[0].sdpo_topk_token_ids = [[0], [42]]
        return 1

    monkeypatch.setattr("prime_rl.orchestrator.sdpo_preflight.wait_for_path", fake_wait_for_path)
    monkeypatch.setattr(
        "prime_rl.orchestrator.sdpo_preflight.load_student_support_records",
        fake_load_student_support_records,
    )
    monkeypatch.setattr(
        "prime_rl.orchestrator.sdpo_preflight.hydrate_student_support_from_records",
        fake_hydrate_student_support_from_records,
    )

    hydrated_rows = asyncio.run(
        run_sdpo_student_support_preflight(
            output_dir=output_dir,
            sender=sender,
            samples=samples,
            step=3,
            expected_topk=2,
        )
    )

    assert hydrated_rows == 1
    assert len(sender.sent) == 1
    sent_batch = sender.sent[0]
    assert sent_batch.examples == [sdpo_sample]
    assert sent_batch.step == 3
    assert sent_batch.preflight_only is True
    assert sdpo_sample.sample_id == "sdpo-preflight-step-3-env-sdpo_env-sample-0"
    assert non_sdpo_sample.sample_id is None
    assert sdpo_sample.sdpo_topk_token_ids == [[0], [42]]


def test_ensure_sdpo_sample_ids_assigns_final_ids_and_preserves_existing_ids():
    teacher_support_sample = SimpleNamespace(
        token_ids=[1],
        mask=[True],
        sdpo_weights=[1.0],
        env_name="sdpo_env",
        sample_id=None,
    )
    student_support_sample = SimpleNamespace(
        token_ids=[2],
        mask=[True],
        sdpo_weights=[1.0],
        env_name="sdpo_env",
        sample_id="sdpo-preflight-step-3-sample-1",
    )
    non_sdpo_sample = SimpleNamespace(token_ids=[3], mask=[True], sdpo_weights=[0.0], sample_id=None)

    ensure_sdpo_sample_ids(
        [teacher_support_sample, student_support_sample, non_sdpo_sample],
        step=3,
        prefix="sdpo-final",
        phase="final",
    )

    assert teacher_support_sample.sample_id == "sdpo-final-step-3-env-sdpo_env-sample-0"
    assert student_support_sample.sample_id == "sdpo-preflight-step-3-sample-1"
    assert non_sdpo_sample.sample_id is None


def test_ensure_sdpo_sample_ids_escapes_generated_env_name_fragment():
    sample = SimpleNamespace(
        token_ids=[1],
        mask=[True],
        sdpo_weights=[1.0],
        env_name="math/env alpha",
        sample_id=None,
    )

    ensure_sdpo_sample_ids([sample], step=4, prefix="sdpo-final", phase="final")

    assert sample.sample_id == "sdpo-final-step-4-env-math%2Fenv%20alpha-sample-0"


def test_ensure_sdpo_sample_ids_rejects_duplicate_final_ids():
    samples = [
        SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[1.0], env_name="sdpo_env", sample_id="duplicate"),
        SimpleNamespace(token_ids=[2], mask=[True], sdpo_weights=[1.0], env_name="sdpo_env", sample_id="duplicate"),
    ]

    with pytest.raises(ValueError, match="duplicate SDPO final sample_id"):
        ensure_sdpo_sample_ids(samples, step=4, prefix="sdpo-final", phase="final")


@pytest.mark.parametrize("sample_id", ["", "   "])
def test_ensure_sdpo_sample_ids_rejects_empty_final_id(sample_id):
    sample = SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[1.0], env_name="sdpo_env", sample_id=sample_id)

    with pytest.raises(ValueError, match="SDPO final sample_id must be non-empty"):
        ensure_sdpo_sample_ids([sample], step=4, prefix="sdpo-final", phase="final")


@pytest.mark.parametrize("env_name", [None, "", "   ", 123])
def test_ensure_sdpo_sample_ids_rejects_missing_env_name(env_name):
    sample = SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[1.0], env_name=env_name, sample_id=None)

    with pytest.raises(ValueError, match="env_name to be a non-empty string"):
        ensure_sdpo_sample_ids([sample], step=4, prefix="sdpo-final", phase="final")


def test_ensure_sdpo_sample_ids_rejects_non_list_sdpo_weights():
    sample = SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=1.0, sample_id=None)

    with pytest.raises(ValueError, match="sdpo_weights to be a list"):
        ensure_sdpo_sample_ids([sample], step=4, prefix="sdpo-final", phase="final")


def test_ensure_sdpo_sample_ids_rejects_boolean_inactive_sdpo_weights():
    sample = SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[False], sample_id=None)

    with pytest.raises(ValueError, match="finite numeric sdpo_weights"):
        ensure_sdpo_sample_ids([sample], step=4, prefix="sdpo-final", phase="final")


@pytest.mark.parametrize("weight", [True, "bad", None, float("nan"), float("inf")])
def test_ensure_sdpo_sample_ids_rejects_non_finite_or_non_numeric_sdpo_weights(weight):
    sample = SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[weight], sample_id=None)

    with pytest.raises(ValueError, match="finite numeric sdpo_weights at token 0"):
        ensure_sdpo_sample_ids([sample], step=4, prefix="sdpo-final", phase="final")


def test_ensure_sdpo_sample_ids_rejects_negative_sdpo_weights():
    sample = SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[-0.5], sample_id=None)

    with pytest.raises(ValueError, match="non-negative sdpo_weights at token 0"):
        ensure_sdpo_sample_ids([sample], step=4, prefix="sdpo-final", phase="final")


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (
            SimpleNamespace(token_ids=(1,), mask=[True], sdpo_weights=[1.0], sample_id=None),
            "token_ids to be a list",
        ),
        (
            SimpleNamespace(token_ids=[1], mask=(True,), sdpo_weights=[1.0], sample_id=None),
            "mask to be a list",
        ),
        (
            SimpleNamespace(token_ids=[1, 2], mask=[True], sdpo_weights=[1.0, 0.0], sample_id=None),
            "mask length to match token_ids length",
        ),
        (
            SimpleNamespace(token_ids=[1, 2], mask=[True, True], sdpo_weights=[1.0], sample_id=None),
            "sdpo_weights length to match token_ids length",
        ),
        (
            SimpleNamespace(token_ids=[1], mask=[1], sdpo_weights=[1.0], sample_id=None),
            "boolean mask values",
        ),
        (
            SimpleNamespace(token_ids=[1], mask=[False], sdpo_weights=[1.0], sample_id=None),
            "zero outside mask",
        ),
    ],
)
def test_ensure_sdpo_sample_ids_rejects_misaligned_sdpo_identity_streams(sample, message):
    with pytest.raises(ValueError, match=message):
        ensure_sdpo_sample_ids([sample], step=4, prefix="sdpo-final", phase="final")


def test_run_sdpo_student_support_preflight_rejects_duplicate_sample_ids(tmp_path, monkeypatch):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    samples = [
        SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[1.0], env_name="sdpo_env", sample_id="duplicate"),
        SimpleNamespace(token_ids=[2], mask=[True], sdpo_weights=[1.0], env_name="sdpo_env", sample_id="duplicate"),
    ]
    export_dir = token_export_step_dir(output_dir, 3)

    async def fake_wait_for_path(path):
        assert path == export_dir / "STABLE"
        path.parent.mkdir(parents=True)
        path.touch()

    monkeypatch.setattr("prime_rl.orchestrator.sdpo_preflight.wait_for_path", fake_wait_for_path)

    with pytest.raises(ValueError, match="duplicate SDPO preflight sample_id"):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=samples,
                step=3,
                expected_topk=2,
            )
        )


def test_run_sdpo_student_support_preflight_rejects_non_string_sample_id(tmp_path, monkeypatch):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    samples = [SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[1.0], env_name="sdpo_env", sample_id=123)]
    export_dir = token_export_step_dir(output_dir, 3)

    async def fake_wait_for_path(path):
        assert path == export_dir / "STABLE"
        path.parent.mkdir(parents=True)
        path.touch()

    monkeypatch.setattr("prime_rl.orchestrator.sdpo_preflight.wait_for_path", fake_wait_for_path)

    with pytest.raises(ValueError, match="SDPO preflight sample_id must be a string"):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=samples,
                step=3,
                expected_topk=2,
            )
        )

    assert sender.sent == []


def test_run_sdpo_student_support_preflight_rejects_boolean_inactive_weights(tmp_path):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    samples = [SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[False], sample_id=None)]

    with pytest.raises(ValueError, match="finite numeric values"):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=samples,
                step=3,
                expected_topk=2,
            )
        )

    assert sender.sent == []


def test_run_sdpo_student_support_preflight_rejects_empty_hydration(tmp_path, monkeypatch):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    samples = [
        SimpleNamespace(
            token_ids=[1, 2],
            mask=[False, True],
            sdpo_weights=[0.0, 1.0],
            env_name="sdpo_env",
            sample_id=None,
        )
    ]
    export_dir = token_export_step_dir(output_dir, 3)

    async def fake_wait_for_path(path):
        assert path == export_dir / "STABLE"
        path.parent.mkdir(parents=True)
        path.touch()

    monkeypatch.setattr("prime_rl.orchestrator.sdpo_preflight.wait_for_path", fake_wait_for_path)
    monkeypatch.setattr(
        "prime_rl.orchestrator.sdpo_preflight.load_student_support_records",
        lambda path, *, require_preflight_only=False: [],
    )
    monkeypatch.setattr(
        "prime_rl.orchestrator.sdpo_preflight.hydrate_student_support_from_records",
        lambda received_samples, records, *, expected_topk=None, require_sample_ids=False: 0,
    )

    with pytest.raises(ValueError, match="exported no usable support rows"):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=samples,
                step=3,
                expected_topk=2,
            )
        )


def test_run_sdpo_student_support_preflight_rejects_no_sdpo_weighted_samples(tmp_path):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    samples = [SimpleNamespace(token_ids=[1, 2], mask=[False, True], sdpo_weights=[0.0, 0.0], sample_id=None)]

    with pytest.raises(ValueError, match="found no SDPO-weighted samples"):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=samples,
                step=3,
                expected_topk=2,
            )
        )

    assert sender.sent == []


def test_run_sdpo_student_support_preflight_requires_expected_topk_before_send(tmp_path):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    export_dir = token_export_step_dir(output_dir, 3)
    export_dir.mkdir(parents=True)
    stale_file = export_dir / "stale.jsonl"
    stale_file.write_text("keep for debugging", encoding="utf-8")
    sample = SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[1.0], env_name="sdpo_env", sample_id=None)

    with pytest.raises(ValueError, match="requires expected_topk"):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=[sample],
                step=3,
            )
        )

    assert sender.sent == []
    assert stale_file.read_text(encoding="utf-8") == "keep for debugging"


@pytest.mark.parametrize("expected_topk", [True, 2.5, "2"])
def test_run_sdpo_student_support_preflight_rejects_non_integer_expected_topk_before_send(tmp_path, expected_topk):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    sample = SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[1.0], env_name="sdpo_env", sample_id=None)

    with pytest.raises(ValueError, match="expected_topk must be an integer"):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=[sample],
                step=3,
                expected_topk=expected_topk,
            )
        )

    assert sender.sent == []


@pytest.mark.parametrize("expected_topk", [0, -1])
def test_run_sdpo_student_support_preflight_rejects_non_positive_expected_topk_before_send(tmp_path, expected_topk):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    sample = SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[1.0], env_name="sdpo_env", sample_id=None)

    with pytest.raises(ValueError, match="expected_topk must be positive"):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=[sample],
                step=3,
                expected_topk=expected_topk,
            )
        )

    assert sender.sent == []


@pytest.mark.parametrize("export_timeout_s", [0, -1])
def test_run_sdpo_student_support_preflight_rejects_non_positive_export_timeout_before_send(tmp_path, export_timeout_s):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    sample = SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[1.0], env_name="sdpo_env", sample_id=None)

    with pytest.raises(ValueError, match="export_timeout_s must be positive"):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=[sample],
                step=3,
                expected_topk=2,
                export_timeout_s=export_timeout_s,
            )
        )

    assert sender.sent == []


@pytest.mark.parametrize("export_timeout_s", [True, "1.0", float("nan")])
def test_run_sdpo_student_support_preflight_rejects_non_numeric_export_timeout_before_send(tmp_path, export_timeout_s):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    sample = SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[1.0], env_name="sdpo_env", sample_id=None)

    with pytest.raises(ValueError, match="export_timeout_s must be a finite numeric value"):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=[sample],
                step=3,
                expected_topk=2,
                export_timeout_s=export_timeout_s,
            )
        )

    assert sender.sent == []


def test_run_sdpo_student_support_preflight_times_out_waiting_for_stable_export(tmp_path, monkeypatch):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    sample = SimpleNamespace(token_ids=[1], mask=[True], sdpo_weights=[1.0], env_name="sdpo_env", sample_id=None)
    export_dir = token_export_step_dir(output_dir, 3)

    async def fake_wait_for_path(path):
        assert path == export_dir / "STABLE"
        await asyncio.sleep(10)

    monkeypatch.setattr("prime_rl.orchestrator.sdpo_preflight.wait_for_path", fake_wait_for_path)

    with pytest.raises(TimeoutError, match="timed out after 0.001s.*SDPO student-support preflight export marker"):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=[sample],
                step=3,
                expected_topk=2,
                export_timeout_s=0.001,
            )
        )

    assert len(sender.sent) == 1
    assert sender.sent[0].preflight_only is True


@pytest.mark.parametrize(
    ("sample_overrides", "message"),
    [
        ({"sdpo_weights": [0.0, True]}, "finite numeric"),
        ({"sdpo_weights": [0.0, float("nan")]}, "finite numeric"),
        ({"sdpo_weights": [0.0, -0.5]}, "non-negative"),
        ({"sdpo_weights": [1.0, 1.0]}, "zero outside sampled tokens"),
        ({"sdpo_weights": [1.0]}, "sdpo_weights length must match token_ids length"),
        ({"sdpo_weights": 1.0}, "sdpo_weights must be a list"),
        ({"mask": [False, 1]}, "mask must contain booleans"),
    ],
)
def test_run_sdpo_student_support_preflight_rejects_malformed_sdpo_weights_before_send(
    tmp_path, sample_overrides, message
):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    export_dir = token_export_step_dir(output_dir, 3)
    export_dir.mkdir(parents=True)
    stale_file = export_dir / "stale.jsonl"
    stale_file.write_text("keep for debugging", encoding="utf-8")
    sample = SimpleNamespace(
        token_ids=[1, 2],
        mask=[False, True],
        sdpo_weights=[0.0, 1.0],
        sample_id=None,
        env_name="sdpo_env",
    )
    for key, value in sample_overrides.items():
        setattr(sample, key, value)

    with pytest.raises(ValueError, match=message):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=[sample],
                step=3,
                expected_topk=2,
            )
        )

    assert sender.sent == []
    assert stale_file.read_text(encoding="utf-8") == "keep for debugging"


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        (1, "token_ids must be a list"),
        ([1, True], "token_ids must contain integers"),
        ([1, "2"], "token_ids must contain integers"),
        ([1, -2], "token_ids must be non-negative"),
    ],
)
def test_run_sdpo_student_support_preflight_rejects_malformed_token_ids_before_send(tmp_path, token_ids, message):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    export_dir = token_export_step_dir(output_dir, 3)
    export_dir.mkdir(parents=True)
    stale_file = export_dir / "stale.jsonl"
    stale_file.write_text("keep for debugging", encoding="utf-8")
    sample = SimpleNamespace(
        token_ids=token_ids,
        mask=[False, True],
        sdpo_weights=[0.0, 1.0],
        sample_id=None,
        env_name="sdpo_env",
    )

    with pytest.raises(ValueError, match=message):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=[sample],
                step=3,
                expected_topk=2,
            )
        )

    assert sender.sent == []
    assert stale_file.read_text(encoding="utf-8") == "keep for debugging"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sdpo_topk_token_ids", [[0, 0], [42, 43]]),
        ("sdpo_topk_logprobs", [[0.0, 0.0], [-0.5, -1.5]]),
        ("sdpo_rollout_is_weights", [0.0, 1.25]),
    ],
)
def test_run_sdpo_student_support_preflight_rejects_preexisting_support_before_send(tmp_path, field, value):
    output_dir = tmp_path / "run_default"
    sender = _FakeSender()
    export_dir = token_export_step_dir(output_dir, 3)
    export_dir.mkdir(parents=True)
    stale_file = export_dir / "stale.jsonl"
    stale_file.write_text("keep for debugging", encoding="utf-8")
    sample = SimpleNamespace(
        token_ids=[1, 2],
        mask=[False, True],
        sdpo_weights=[0.0, 1.0],
        sdpo_topk_token_ids=None,
        sdpo_topk_logprobs=None,
        sdpo_rollout_is_weights=None,
        sample_id=None,
        env_name="sdpo_env",
    )
    setattr(sample, field, value)

    with pytest.raises(ValueError, match=f"pre-existing {field}"):
        asyncio.run(
            run_sdpo_student_support_preflight(
                output_dir=output_dir,
                sender=sender,
                samples=[sample],
                step=3,
                expected_topk=2,
            )
        )

    assert sender.sent == []
    assert stale_file.read_text(encoding="utf-8") == "keep for debugging"
