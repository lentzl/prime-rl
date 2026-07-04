from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.types import Policy
from prime_rl.orchestrator.watcher import InferenceWeightTarget, WeightWatcher


class RecordingInferencePool:
    def __init__(self, role: str, events: list[tuple], *, fail_update: bool = False):
        self.role = role
        self.events = events
        self.model_name = role
        self.fail_update = fail_update

    async def update_weights(self, weight_dir, lora_name=None, step=0):
        if self.fail_update:
            self.events.append(("update_weights_failed", self.role, weight_dir, lora_name, step))
            raise RuntimeError(f"{self.role} update failed")
        self.events.append(("update_weights", self.role, weight_dir, lora_name, step))

    def update_model_name(self, model_name: str) -> None:
        self.model_name = model_name
        self.events.append(("update_model_name", self.role, model_name))


class RecordingObserver:
    def __init__(self, events: list[tuple]):
        self.events = events

    async def on_version_pending(self, step: int) -> None:
        self.events.append(("pending", step))

    async def on_new_version(self, step: int) -> None:
        self.events.append(("new_version", step))


@pytest.mark.asyncio
async def test_weight_watcher_updates_extra_inference_targets_before_policy_endpoint_and_new_version(tmp_path):
    events = []
    step_path = tmp_path / "broadcasts" / "step_1"
    step_path.mkdir(parents=True)
    (step_path / "STABLE").touch()
    teacher_path = step_path / "sdpo_teacher"
    teacher_path.mkdir()
    (teacher_path / "STABLE").touch()
    policy = Policy(version=0, model_name="policy")
    policy_pool = RecordingInferencePool("policy", events)
    teacher_pool = RecordingInferencePool("sdpo_teacher", events)
    observer = RecordingObserver(events)
    watcher = WeightWatcher(
        SimpleNamespace(output_dir=tmp_path),
        policy=policy,
        inference=policy_pool,
        observers=[observer],
        lora_name=None,
        extra_inference_targets=[
            InferenceWeightTarget(
                role="sdpo_teacher",
                inference=teacher_pool,
                weight_subdir="sdpo_teacher",
                model_name="sdpo-teacher",
            )
        ],
    )

    await watcher.apply_policy_update(1)

    assert policy.version == 1
    assert watcher.ckpt_step == 1
    assert events == [
        ("pending", 1),
        ("update_weights", "sdpo_teacher", teacher_path, None, 1),
        ("update_model_name", "sdpo_teacher", "sdpo-teacher"),
        ("update_weights", "policy", step_path, None, 1),
        ("new_version", 1),
    ]


@pytest.mark.asyncio
async def test_weight_watcher_waits_for_extra_target_before_any_weight_update(monkeypatch, tmp_path):
    events = []
    step_path = tmp_path / "broadcasts" / "step_1"
    step_path.mkdir(parents=True)
    (step_path / "STABLE").touch()
    teacher_path = step_path / "sdpo_teacher"
    teacher_path.mkdir()
    teacher_stable = teacher_path / "STABLE"

    async def fake_wait_for_path(path):
        events.append(("wait_for_path", path))
        path.touch()

    monkeypatch.setattr("prime_rl.orchestrator.watcher.wait_for_path", fake_wait_for_path)

    policy = Policy(version=0, model_name="policy")
    watcher = WeightWatcher(
        SimpleNamespace(output_dir=tmp_path),
        policy=policy,
        inference=RecordingInferencePool("policy", events),
        observers=[RecordingObserver(events)],
        lora_name=None,
        extra_inference_targets=[
            InferenceWeightTarget(
                role="sdpo_teacher",
                inference=RecordingInferencePool("sdpo_teacher", events),
                weight_subdir="sdpo_teacher",
            )
        ],
    )

    await watcher.apply_policy_update(1)

    assert teacher_stable.exists()
    assert policy.version == 1
    assert events == [
        ("wait_for_path", teacher_stable),
        ("pending", 1),
        ("update_weights", "sdpo_teacher", teacher_path, None, 1),
        ("update_weights", "policy", step_path, None, 1),
        ("new_version", 1),
    ]


@pytest.mark.asyncio
async def test_weight_watcher_does_not_advance_policy_endpoint_when_extra_target_update_fails(tmp_path):
    events = []
    step_path = tmp_path / "broadcasts" / "step_1"
    step_path.mkdir(parents=True)
    (step_path / "STABLE").touch()
    teacher_path = step_path / "sdpo_teacher"
    teacher_path.mkdir()
    (teacher_path / "STABLE").touch()
    policy = Policy(version=0, model_name="policy")
    watcher = WeightWatcher(
        SimpleNamespace(output_dir=tmp_path),
        policy=policy,
        inference=RecordingInferencePool("policy", events),
        observers=[RecordingObserver(events)],
        lora_name=None,
        extra_inference_targets=[
            InferenceWeightTarget(
                role="sdpo_teacher",
                inference=RecordingInferencePool("sdpo_teacher", events, fail_update=True),
                weight_subdir="sdpo_teacher",
            )
        ],
    )

    with pytest.raises(RuntimeError, match="sdpo_teacher update failed"):
        await watcher.apply_policy_update(1)

    assert policy.version == 0
    assert watcher.ckpt_step == 0
    assert events == [
        ("pending", 1),
        ("update_weights_failed", "sdpo_teacher", teacher_path, None, 1),
    ]
