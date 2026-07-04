from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful

from prime_rl.trainer.rl.sdpo_teacher import (
    SDPO_TEACHER_BROADCAST_ROLE,
    SDPOEMATeacher,
    SDPOTeacherRegularizationSetup,
    SDPOTrustRegionTeacher,
    sdpo_optimizer_update_succeeded,
    sdpo_teacher_broadcast_models,
    setup_sdpo_teacher_regularization,
    setup_sdpo_teacher_regularization_from_runtime,
    step_sdpo_ema_teacher_if_updated,
    step_sdpo_teacher_regularization_if_updated,
    sync_sdpo_teacher_from_student_,
    update_sdpo_ema_teacher_,
)


class ModuleWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.register_buffer("scale", torch.tensor([1.0]))


class SharedParameterWrapper(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.weight = parameter


class SharedBufferWrapper(nn.Module):
    def __init__(self, buffer):
        super().__init__()
        self.register_buffer("scale", buffer)


class LogitsModule(nn.Module):
    def __init__(self, logits, *, as_tuple=False):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))
        self.as_tuple = as_tuple

    def forward(self, *args, **kwargs):
        if self.as_tuple:
            return (self.logits,)
        return SimpleNamespace(logits=self.logits)


class RecordingLogitsModule(LogitsModule):
    def __init__(self, logits):
        super().__init__(logits)
        self.calls = []

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return super().forward(*args, **kwargs)


class ParameterizedLogitsModule(LogitsModule):
    def __init__(self, logits):
        super().__init__(logits)
        self.param = nn.Parameter(torch.tensor([1.0]))


def test_sync_sdpo_teacher_from_student_copies_parameters_and_buffers():
    teacher = ModuleWithBuffer()
    student = ModuleWithBuffer()
    with torch.no_grad():
        teacher.linear.weight.zero_()
        teacher.linear.bias.zero_()
        teacher.scale.zero_()
        student.linear.weight.copy_(torch.tensor([[2.0, 4.0]]))
        student.linear.bias.copy_(torch.tensor([6.0]))
        student.scale.copy_(torch.tensor([8.0]))

    sync_sdpo_teacher_from_student_(teacher, student)

    torch.testing.assert_close(teacher.linear.weight, student.linear.weight)
    torch.testing.assert_close(teacher.linear.bias, student.linear.bias)
    torch.testing.assert_close(teacher.scale, student.scale)


def test_sdpo_ema_teacher_syncs_and_steps_bound_modules():
    teacher = nn.Linear(2, 1)
    student = nn.Linear(2, 1)
    with torch.no_grad():
        teacher.weight.zero_()
        teacher.bias.zero_()
        student.weight.copy_(torch.tensor([[2.0, 4.0]]))
        student.bias.copy_(torch.tensor([6.0]))

    ema_teacher = SDPOEMATeacher(teacher, student, update_rate=0.5)
    ema_teacher.sync_from_student()

    torch.testing.assert_close(teacher.weight, student.weight)
    torch.testing.assert_close(teacher.bias, student.bias)

    with torch.no_grad():
        student.weight.copy_(torch.tensor([[4.0, 8.0]]))
        student.bias.copy_(torch.tensor([10.0]))

    ema_teacher.step()

    torch.testing.assert_close(teacher.weight, torch.tensor([[3.0, 6.0]]))
    torch.testing.assert_close(teacher.bias, torch.tensor([8.0]))


def test_sdpo_ema_teacher_is_checkpoint_stateful():
    ema_teacher = SDPOEMATeacher(nn.Linear(2, 1), nn.Linear(2, 1), update_rate=0.5)

    assert isinstance(ema_teacher, Stateful)
    assert ema_teacher.requires_checkpoint_state


def test_sdpo_ema_teacher_freezes_teacher_but_allows_explicit_updates():
    teacher = nn.Linear(2, 1)
    student = nn.Linear(2, 1)
    teacher.train()
    with torch.no_grad():
        teacher.weight.zero_()
        teacher.bias.zero_()
        student.weight.copy_(torch.tensor([[2.0, 4.0]]))
        student.bias.copy_(torch.tensor([6.0]))

    ema_teacher = SDPOEMATeacher(teacher, student, update_rate=0.5)

    assert not ema_teacher.teacher.training
    assert all(not param.requires_grad for param in ema_teacher.teacher.parameters())

    ema_teacher.step()

    torch.testing.assert_close(teacher.weight, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(teacher.bias, torch.tensor([3.0]))


def test_sdpo_ema_teacher_state_dict_round_trips_teacher_state_and_rate():
    teacher = ModuleWithBuffer()
    student = ModuleWithBuffer()
    with torch.no_grad():
        teacher.linear.weight.copy_(torch.tensor([[2.0, 4.0]]))
        teacher.linear.bias.copy_(torch.tensor([6.0]))
        teacher.scale.copy_(torch.tensor([8.0]))

    ema_teacher = SDPOEMATeacher(teacher, student, update_rate=0.25)
    state_dict = ema_teacher.state_dict()

    next_teacher = ModuleWithBuffer()
    next_student = ModuleWithBuffer()
    next_ema_teacher = SDPOEMATeacher(next_teacher, next_student, update_rate=0.75)
    next_ema_teacher.load_state_dict(state_dict)

    assert next_ema_teacher.update_rate == 0.25
    torch.testing.assert_close(next_teacher.linear.weight, teacher.linear.weight)
    torch.testing.assert_close(next_teacher.linear.bias, teacher.linear.bias)
    torch.testing.assert_close(next_teacher.scale, teacher.scale)


def test_sdpo_ema_teacher_load_state_dict_rejects_invalid_rate():
    ema_teacher = SDPOEMATeacher(ModuleWithBuffer(), ModuleWithBuffer(), update_rate=0.25)
    state_dict = ema_teacher.state_dict()
    state_dict["update_rate"] = 2.0

    with pytest.raises(ValueError, match="update_rate"):
        ema_teacher.load_state_dict(state_dict)


def test_step_sdpo_ema_teacher_if_updated_steps_after_successful_optimizer_update():
    teacher = nn.Linear(2, 1)
    student = nn.Linear(2, 1)
    with torch.no_grad():
        teacher.weight.zero_()
        teacher.bias.zero_()
        student.weight.copy_(torch.tensor([[2.0, 4.0]]))
        student.bias.copy_(torch.tensor([6.0]))
    ema_teacher = SDPOEMATeacher(teacher, student, update_rate=0.5)

    did_step = step_sdpo_ema_teacher_if_updated(ema_teacher, update_succeeded=True)

    assert did_step
    torch.testing.assert_close(teacher.weight, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(teacher.bias, torch.tensor([3.0]))


def test_step_sdpo_ema_teacher_if_updated_skips_failed_optimizer_update():
    teacher = nn.Linear(2, 1)
    student = nn.Linear(2, 1)
    with torch.no_grad():
        teacher.weight.zero_()
        teacher.bias.zero_()
        student.weight.copy_(torch.tensor([[2.0, 4.0]]))
        student.bias.copy_(torch.tensor([6.0]))
    ema_teacher = SDPOEMATeacher(teacher, student, update_rate=0.5)

    did_step = step_sdpo_ema_teacher_if_updated(ema_teacher, update_succeeded=False)

    assert not did_step
    torch.testing.assert_close(teacher.weight, torch.zeros_like(teacher.weight))
    torch.testing.assert_close(teacher.bias, torch.zeros_like(teacher.bias))


def test_step_sdpo_teacher_regularization_if_updated_steps_ema_extra_state():
    teacher = nn.Linear(2, 1)
    student = nn.Linear(2, 1)
    with torch.no_grad():
        teacher.weight.zero_()
        teacher.bias.zero_()
        student.weight.copy_(torch.tensor([[2.0, 4.0]]))
        student.bias.copy_(torch.tensor([6.0]))
    setup = SDPOTeacherRegularizationSetup(
        scoring_module=teacher,
        extra_state=SDPOEMATeacher(teacher, student, update_rate=0.5),
    )

    did_step = step_sdpo_teacher_regularization_if_updated(setup, update_succeeded=True)

    assert did_step
    torch.testing.assert_close(teacher.weight, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(teacher.bias, torch.tensor([3.0]))


def test_step_sdpo_teacher_regularization_if_updated_skips_without_ema_state():
    setup = SDPOTeacherRegularizationSetup(scoring_module=nn.Linear(2, 1))

    assert not step_sdpo_teacher_regularization_if_updated(setup, update_succeeded=True)
    assert not step_sdpo_teacher_regularization_if_updated(None, update_succeeded=True)


def test_sdpo_teacher_broadcast_models_returns_ema_teacher_module():
    teacher = nn.Linear(2, 1)
    student = nn.Linear(2, 1)
    setup = SDPOTeacherRegularizationSetup(
        scoring_module=teacher,
        extra_state=SDPOEMATeacher(teacher, student, update_rate=0.5),
    )

    assert sdpo_teacher_broadcast_models(setup) == {SDPO_TEACHER_BROADCAST_ROLE: teacher}


def test_sdpo_teacher_broadcast_models_uses_ema_owned_teacher_module():
    teacher = nn.Linear(2, 1)
    student = nn.Linear(2, 1)
    stale_scoring_module = nn.Linear(2, 1)
    setup = SDPOTeacherRegularizationSetup(
        scoring_module=stale_scoring_module,
        extra_state=SDPOEMATeacher(teacher, student, update_rate=0.5),
    )

    assert sdpo_teacher_broadcast_models(setup) == {SDPO_TEACHER_BROADCAST_ROLE: teacher}


def test_sdpo_teacher_broadcast_models_skips_absent_and_non_ema_state():
    assert sdpo_teacher_broadcast_models(None) == {}
    assert sdpo_teacher_broadcast_models(SDPOTeacherRegularizationSetup(scoring_module=nn.Linear(2, 1))) == {}

    reference = LogitsModule([[1.0]])
    student = LogitsModule([[2.0]])
    setup = setup_sdpo_teacher_regularization(
        "trust-region",
        student_module=student,
        reference_module=reference,
        teacher_update_rate=0.5,
    )
    assert sdpo_teacher_broadcast_models(setup) == {}


def test_sdpo_optimizer_update_succeeded_treats_absent_grad_norm_as_success():
    assert sdpo_optimizer_update_succeeded(None)


def test_sdpo_optimizer_update_succeeded_checks_grad_norm_finiteness():
    assert sdpo_optimizer_update_succeeded(torch.tensor(1.0))
    assert not sdpo_optimizer_update_succeeded(torch.tensor(float("nan")))
    assert not sdpo_optimizer_update_succeeded(torch.tensor(float("inf")))


def test_setup_sdpo_teacher_regularization_live_policy_uses_student_module():
    student = nn.Linear(2, 1)

    setup = setup_sdpo_teacher_regularization(
        "live-policy",
        student_module=student,
        teacher_update_rate=0.05,
    )

    assert setup.scoring_module is student
    assert setup.extra_state is None


def test_setup_sdpo_teacher_regularization_ema_returns_checkpointable_owner():
    teacher = nn.Linear(2, 1)
    student = nn.Linear(2, 1)
    with torch.no_grad():
        teacher.weight.zero_()
        teacher.bias.zero_()
        student.weight.copy_(torch.tensor([[2.0, 4.0]]))
        student.bias.copy_(torch.tensor([6.0]))

    setup = setup_sdpo_teacher_regularization(
        "ema",
        student_module=student,
        teacher_module=teacher,
        teacher_update_rate=0.05,
    )

    assert setup.scoring_module is teacher
    assert isinstance(setup.extra_state, SDPOEMATeacher)
    assert setup.extra_state.teacher is teacher
    assert setup.extra_state.student is student
    torch.testing.assert_close(teacher.weight, student.weight)
    torch.testing.assert_close(teacher.bias, student.bias)


def test_setup_sdpo_teacher_regularization_from_runtime_builds_ema_teacher():
    teacher = nn.Linear(2, 1)
    student = nn.Linear(2, 1)
    with torch.no_grad():
        teacher.weight.zero_()
        teacher.bias.zero_()
        student.weight.copy_(torch.tensor([[2.0, 4.0]]))
        student.bias.copy_(torch.tensor([6.0]))

    setup = setup_sdpo_teacher_regularization_from_runtime(
        "ema",
        student_module=student,
        teacher_update_rate=0.25,
        teacher_module_factory=lambda: teacher,
    )

    assert setup.scoring_module is teacher
    assert isinstance(setup.extra_state, SDPOEMATeacher)
    assert setup.extra_state.update_rate == 0.25
    torch.testing.assert_close(teacher.weight, student.weight)
    torch.testing.assert_close(teacher.bias, student.bias)


def test_setup_sdpo_teacher_regularization_from_runtime_requires_factories():
    with pytest.raises(NotImplementedError, match="teacher_module_factory"):
        setup_sdpo_teacher_regularization_from_runtime(
            "ema",
            student_module=nn.Linear(2, 1),
            teacher_update_rate=0.05,
        )

    with pytest.raises(NotImplementedError, match="reference_module_factory"):
        setup_sdpo_teacher_regularization_from_runtime(
            "trust-region",
            student_module=nn.Linear(2, 1),
            teacher_update_rate=0.05,
        )


def test_setup_sdpo_teacher_regularization_trust_region_returns_mixed_teacher():
    reference = LogitsModule([[1.0, 3.0]])
    student = LogitsModule([[5.0, 7.0]])

    setup = setup_sdpo_teacher_regularization(
        "trust-region",
        student_module=student,
        reference_module=reference,
        teacher_update_rate=0.25,
    )

    assert isinstance(setup.scoring_module, SDPOTrustRegionTeacher)
    assert setup.extra_state is None
    torch.testing.assert_close(setup.scoring_module().logits, torch.tensor([[2.0, 4.0]]))


def test_setup_sdpo_teacher_regularization_requires_teacher_modules():
    student = nn.Linear(2, 1)

    with pytest.raises(ValueError, match="requires a separate teacher_module"):
        setup_sdpo_teacher_regularization("ema", student_module=student, teacher_update_rate=0.05)

    with pytest.raises(ValueError, match="requires a separate reference_module"):
        setup_sdpo_teacher_regularization("trust-region", student_module=student, teacher_update_rate=0.05)


def test_setup_sdpo_teacher_regularization_rejects_invalid_mode():
    with pytest.raises(ValueError, match="unsupported SDPO teacher_regularization"):
        setup_sdpo_teacher_regularization("frozen", student_module=nn.Linear(2, 1), teacher_update_rate=0.05)


def test_sdpo_trust_region_teacher_mixes_reference_and_student_logits():
    reference = LogitsModule([[1.0, 3.0]])
    student = LogitsModule([[5.0, 7.0]])

    teacher = SDPOTrustRegionTeacher(reference, student, mix_coef=0.25)
    output = teacher()

    torch.testing.assert_close(output.logits, torch.tensor([[2.0, 4.0]]))


def test_sdpo_trust_region_teacher_accepts_tuple_outputs():
    reference = LogitsModule([[0.0, 2.0]], as_tuple=True)
    student = LogitsModule([[4.0, 6.0]], as_tuple=True)

    teacher = SDPOTrustRegionTeacher(reference, student, mix_coef=0.5)
    output = teacher()

    torch.testing.assert_close(output.logits, torch.tensor([[2.0, 4.0]]))


def test_sdpo_trust_region_teacher_forwards_call_arguments_to_both_modules():
    reference = RecordingLogitsModule([[1.0]])
    student = RecordingLogitsModule([[3.0]])
    teacher = SDPOTrustRegionTeacher(reference, student, mix_coef=0.5)

    output = teacher("tokens", attention_mask="mask")

    torch.testing.assert_close(output.logits, torch.tensor([[2.0]]))
    assert reference.calls == [(("tokens",), {"attention_mask": "mask"})]
    assert student.calls == [(("tokens",), {"attention_mask": "mask"})]


def test_sdpo_trust_region_teacher_freezes_reference_only():
    reference = ParameterizedLogitsModule([[1.0]])
    student = ParameterizedLogitsModule([[3.0]])
    reference.train()
    student.train()

    teacher = SDPOTrustRegionTeacher(reference, student, mix_coef=0.5)

    assert not teacher.reference.training
    assert teacher.student.training
    assert all(not param.requires_grad for param in teacher.reference.parameters())
    assert all(param.requires_grad for param in teacher.student.parameters())


def test_update_sdpo_ema_teacher_updates_parameters_in_place():
    teacher = nn.Linear(2, 1)
    student = nn.Linear(2, 1)
    with torch.no_grad():
        teacher.weight.copy_(torch.tensor([[1.0, 3.0]]))
        teacher.bias.copy_(torch.tensor([5.0]))
        student.weight.copy_(torch.tensor([[5.0, 7.0]]))
        student.bias.copy_(torch.tensor([1.0]))

    update_sdpo_ema_teacher_(teacher, student, update_rate=0.25)

    torch.testing.assert_close(teacher.weight, torch.tensor([[2.0, 4.0]]))
    torch.testing.assert_close(teacher.bias, torch.tensor([4.0]))


def test_update_sdpo_ema_teacher_copies_student_buffers():
    teacher = ModuleWithBuffer()
    student = ModuleWithBuffer()
    with torch.no_grad():
        teacher.linear.weight.copy_(torch.tensor([[1.0, 3.0]]))
        teacher.linear.bias.copy_(torch.tensor([5.0]))
        teacher.scale.copy_(torch.tensor([7.0]))
        student.linear.weight.copy_(torch.tensor([[5.0, 7.0]]))
        student.linear.bias.copy_(torch.tensor([1.0]))
        student.scale.copy_(torch.tensor([11.0]))

    update_sdpo_ema_teacher_(teacher, student, update_rate=0.25)

    torch.testing.assert_close(teacher.linear.weight, torch.tensor([[2.0, 4.0]]))
    torch.testing.assert_close(teacher.linear.bias, torch.tensor([4.0]))
    torch.testing.assert_close(teacher.scale, torch.tensor([11.0]))


def test_update_sdpo_ema_teacher_one_copies_student_parameters():
    teacher = nn.Linear(2, 1)
    student = nn.Linear(2, 1)
    with torch.no_grad():
        teacher.weight.zero_()
        teacher.bias.zero_()
        student.weight.copy_(torch.tensor([[2.0, 4.0]]))
        student.bias.copy_(torch.tensor([6.0]))

    update_sdpo_ema_teacher_(teacher, student, update_rate=1.0)

    torch.testing.assert_close(teacher.weight, student.weight)
    torch.testing.assert_close(teacher.bias, student.bias)


def test_update_sdpo_ema_teacher_copies_student_buffers_after_parameter_ema():
    teacher = ModuleWithBuffer()
    student = ModuleWithBuffer()
    with torch.no_grad():
        teacher.linear.weight.zero_()
        teacher.linear.bias.zero_()
        teacher.scale.copy_(torch.tensor([1.0]))
        student.linear.weight.copy_(torch.tensor([[2.0, 4.0]]))
        student.linear.bias.copy_(torch.tensor([6.0]))
        student.scale.copy_(torch.tensor([8.0]))

    update_sdpo_ema_teacher_(teacher, student, update_rate=0.5)

    torch.testing.assert_close(teacher.linear.weight, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(teacher.linear.bias, torch.tensor([3.0]))
    torch.testing.assert_close(teacher.scale, torch.tensor([8.0]))


def test_update_sdpo_ema_teacher_zero_leaves_teacher_unchanged():
    teacher = ModuleWithBuffer()
    student = ModuleWithBuffer()
    before = {name: param.detach().clone() for name, param in teacher.named_parameters()}
    before_buffers = {name: buffer.detach().clone() for name, buffer in teacher.named_buffers()}
    with torch.no_grad():
        student.scale.copy_(torch.tensor([8.0]))

    update_sdpo_ema_teacher_(teacher, student, update_rate=0.0)

    for name, param in teacher.named_parameters():
        torch.testing.assert_close(param, before[name])
    for name, buffer in teacher.named_buffers():
        torch.testing.assert_close(buffer, before_buffers[name])


def test_update_sdpo_ema_teacher_rejects_shared_module():
    module = nn.Linear(2, 1)

    with pytest.raises(ValueError, match="separate"):
        update_sdpo_ema_teacher_(module, module, update_rate=0.5)


def test_update_sdpo_ema_teacher_rejects_shared_parameter_object():
    parameter = nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(ValueError, match="share parameter objects"):
        update_sdpo_ema_teacher_(
            SharedParameterWrapper(parameter),
            SharedParameterWrapper(parameter),
            update_rate=0.5,
        )


def test_update_sdpo_ema_teacher_rejects_shared_buffer_object():
    buffer = torch.tensor([1.0])

    with pytest.raises(ValueError, match="share buffer objects"):
        update_sdpo_ema_teacher_(
            SharedBufferWrapper(buffer),
            SharedBufferWrapper(buffer),
            update_rate=0.5,
        )


def test_sync_sdpo_teacher_from_student_rejects_shared_module():
    module = nn.Linear(2, 1)

    with pytest.raises(ValueError, match="separate"):
        sync_sdpo_teacher_from_student_(module, module)


def test_update_sdpo_ema_teacher_rejects_invalid_rate():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        update_sdpo_ema_teacher_(nn.Linear(2, 1), nn.Linear(2, 1), update_rate=1.1)


@pytest.mark.parametrize("update_rate", [True, "0.5", float("nan")])
def test_update_sdpo_ema_teacher_rejects_non_numeric_rates(update_rate):
    with pytest.raises(ValueError, match="finite numeric value"):
        update_sdpo_ema_teacher_(nn.Linear(2, 1), nn.Linear(2, 1), update_rate=update_rate)


def test_sdpo_ema_teacher_rejects_invalid_rate():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SDPOEMATeacher(nn.Linear(2, 1), nn.Linear(2, 1), update_rate=-0.1)


@pytest.mark.parametrize("update_rate", [True, "0.5", float("nan")])
def test_sdpo_ema_teacher_rejects_non_numeric_constructor_rates(update_rate):
    with pytest.raises(ValueError, match="finite numeric value"):
        SDPOEMATeacher(nn.Linear(2, 1), nn.Linear(2, 1), update_rate=update_rate)


@pytest.mark.parametrize("update_rate", [True, "0.5", float("nan")])
def test_sdpo_ema_teacher_load_state_dict_rejects_non_numeric_rates(update_rate):
    ema_teacher = SDPOEMATeacher(ModuleWithBuffer(), ModuleWithBuffer(), update_rate=0.25)
    state_dict = ema_teacher.state_dict()
    state_dict["update_rate"] = update_rate

    with pytest.raises(ValueError, match="finite numeric value"):
        ema_teacher.load_state_dict(state_dict)


@pytest.mark.parametrize("teacher_update_rate", [True, "0.5", float("nan")])
def test_setup_sdpo_teacher_regularization_rejects_non_numeric_rates(teacher_update_rate):
    with pytest.raises(ValueError, match="finite numeric value"):
        setup_sdpo_teacher_regularization(
            "live-policy",
            student_module=nn.Linear(2, 1),
            teacher_update_rate=teacher_update_rate,
        )


def test_sdpo_trust_region_teacher_rejects_invalid_mix_coef():
    with pytest.raises(ValueError, match=r"mix_coef must be in \[0, 1\]"):
        SDPOTrustRegionTeacher(LogitsModule([[1.0]]), LogitsModule([[2.0]]), mix_coef=1.1)


@pytest.mark.parametrize("mix_coef", [True, "0.5", float("nan")])
def test_sdpo_trust_region_teacher_rejects_non_numeric_mix_coef(mix_coef):
    with pytest.raises(ValueError, match="finite numeric value"):
        SDPOTrustRegionTeacher(LogitsModule([[1.0]]), LogitsModule([[2.0]]), mix_coef=mix_coef)


def test_sdpo_trust_region_teacher_rejects_shared_module():
    module = LogitsModule([[1.0]])

    with pytest.raises(ValueError, match="trust-region teacher"):
        SDPOTrustRegionTeacher(module, module, mix_coef=0.5)


def test_update_sdpo_ema_teacher_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="parameter shapes differ"):
        update_sdpo_ema_teacher_(nn.Linear(2, 1), nn.Linear(3, 1), update_rate=0.5)


def test_update_sdpo_ema_teacher_rejects_parameter_name_mismatch():
    with pytest.raises(ValueError, match="parameter names differ"):
        update_sdpo_ema_teacher_(nn.Linear(2, 1), nn.Sequential(nn.Linear(2, 1)), update_rate=0.5)


def test_update_sdpo_ema_teacher_rejects_buffer_shape_mismatch():
    teacher = ModuleWithBuffer()
    student = ModuleWithBuffer()
    student.scale = torch.ones(2)

    with pytest.raises(ValueError, match="buffer shapes differ"):
        update_sdpo_ema_teacher_(teacher, student, update_rate=0.5)


def test_update_sdpo_ema_teacher_rejects_buffer_name_mismatch():
    teacher = ModuleWithBuffer()
    student = ModuleWithBuffer()
    del student.scale
    student.register_buffer("other_scale", torch.tensor([1.0]))

    with pytest.raises(ValueError, match="buffer names differ"):
        update_sdpo_ema_teacher_(teacher, student, update_rate=0.5)


def test_sync_sdpo_teacher_from_student_rejects_state_key_mismatch():
    with pytest.raises(ValueError, match="state_dict keys differ"):
        sync_sdpo_teacher_from_student_(nn.Linear(2, 1), nn.Sequential(nn.Linear(2, 1)))


def test_sdpo_trust_region_teacher_rejects_logits_shape_mismatch():
    teacher = SDPOTrustRegionTeacher(LogitsModule([[1.0, 2.0]]), LogitsModule([[3.0]]), mix_coef=0.5)

    with pytest.raises(ValueError, match="logits shapes differ"):
        teacher()
