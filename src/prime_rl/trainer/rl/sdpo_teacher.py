from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful

from prime_rl.trainer.optim import optimizer_update_succeeded

SDPOTeacherRegularization = Literal["live-policy", "ema", "trust-region"]
SDPO_TEACHER_BROADCAST_ROLE = "sdpo_teacher"


@dataclass(frozen=True)
class SDPOTeacherRegularizationSetup:
    scoring_module: nn.Module
    extra_state: Stateful | None = None


def _validate_unit_interval(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite numeric value in [0, 1], got {value!r}")
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return normalized


def _validate_distinct_modules(first: nn.Module, second: nn.Module, *, context: str) -> None:
    if first is second:
        raise ValueError(f"{context} must use separate modules")
    first_parameter_ids = {id(parameter) for parameter in first.parameters()}
    second_parameter_ids = {id(parameter) for parameter in second.parameters()}
    if first_parameter_ids & second_parameter_ids:
        raise ValueError(f"{context} must not share parameter objects")
    first_buffer_ids = {id(buffer) for buffer in first.buffers()}
    second_buffer_ids = {id(buffer) for buffer in second.buffers()}
    if first_buffer_ids & second_buffer_ids:
        raise ValueError(f"{context} must not share buffer objects")


class SDPOEMATeacher(Stateful):
    requires_checkpoint_state = True

    def __init__(self, teacher: nn.Module, student: nn.Module, update_rate: float):
        update_rate = _validate_unit_interval(update_rate, name="update_rate")
        _validate_distinct_modules(teacher, student, context="SDPO EMA teacher")
        self.teacher = teacher
        self.student = student
        self.update_rate = update_rate
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def state_dict(self) -> dict[str, Any]:
        return {
            "update_rate": self.update_rate,
            "teacher": self.teacher.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        update_rate = _validate_unit_interval(state_dict["update_rate"], name="update_rate")
        self.teacher.load_state_dict(state_dict["teacher"])
        self.update_rate = update_rate

    def sync_from_student(self) -> None:
        sync_sdpo_teacher_from_student_(self.teacher, self.student)

    def step(self) -> None:
        update_sdpo_ema_teacher_(self.teacher, self.student, self.update_rate)


def step_sdpo_ema_teacher_if_updated(ema_teacher: SDPOEMATeacher, *, update_succeeded: bool) -> bool:
    if not update_succeeded:
        return False
    ema_teacher.step()
    return True


def step_sdpo_teacher_regularization_if_updated(
    setup: SDPOTeacherRegularizationSetup | None, *, update_succeeded: bool
) -> bool:
    if setup is None or setup.extra_state is None:
        return False
    if isinstance(setup.extra_state, SDPOEMATeacher):
        return step_sdpo_ema_teacher_if_updated(setup.extra_state, update_succeeded=update_succeeded)
    return False


def sdpo_teacher_broadcast_models(setup: SDPOTeacherRegularizationSetup | None) -> dict[str, nn.Module]:
    if setup is None or setup.extra_state is None:
        return {}
    if isinstance(setup.extra_state, SDPOEMATeacher):
        return {SDPO_TEACHER_BROADCAST_ROLE: setup.extra_state.teacher}
    return {}


def sdpo_optimizer_update_succeeded(grad_norm: torch.Tensor | None) -> bool:
    return optimizer_update_succeeded(grad_norm)


def setup_sdpo_teacher_regularization(
    teacher_regularization: SDPOTeacherRegularization,
    *,
    student_module: nn.Module,
    teacher_update_rate: float,
    teacher_module: nn.Module | None = None,
    reference_module: nn.Module | None = None,
) -> SDPOTeacherRegularizationSetup:
    teacher_update_rate = _validate_unit_interval(teacher_update_rate, name="teacher_update_rate")
    if teacher_regularization == "live-policy":
        return SDPOTeacherRegularizationSetup(scoring_module=student_module)
    if teacher_regularization == "ema":
        if teacher_module is None:
            raise ValueError("SDPO EMA teacher_regularization requires a separate teacher_module")
        ema_teacher = SDPOEMATeacher(teacher_module, student_module, teacher_update_rate)
        ema_teacher.sync_from_student()
        return SDPOTeacherRegularizationSetup(scoring_module=teacher_module, extra_state=ema_teacher)
    if teacher_regularization == "trust-region":
        if reference_module is None:
            raise ValueError("SDPO trust-region teacher_regularization requires a separate reference_module")
        return SDPOTeacherRegularizationSetup(
            scoring_module=SDPOTrustRegionTeacher(reference_module, student_module, teacher_update_rate)
        )
    raise ValueError(f"unsupported SDPO teacher_regularization: {teacher_regularization!r}")


def setup_sdpo_teacher_regularization_from_runtime(
    teacher_regularization: SDPOTeacherRegularization,
    *,
    student_module: nn.Module,
    teacher_update_rate: float,
    teacher_module_factory: Callable[[], nn.Module] | None = None,
    reference_module_factory: Callable[[], nn.Module] | None = None,
) -> SDPOTeacherRegularizationSetup:
    if teacher_regularization == "live-policy":
        return setup_sdpo_teacher_regularization(
            teacher_regularization,
            student_module=student_module,
            teacher_update_rate=teacher_update_rate,
        )
    if teacher_regularization == "ema":
        if teacher_module_factory is None:
            raise NotImplementedError("SDPO EMA teacher_regularization requires a teacher_module_factory.")
        return setup_sdpo_teacher_regularization(
            teacher_regularization,
            student_module=student_module,
            teacher_module=teacher_module_factory(),
            teacher_update_rate=teacher_update_rate,
        )
    if teacher_regularization == "trust-region":
        if reference_module_factory is None:
            raise NotImplementedError("SDPO trust-region teacher_regularization requires a reference_module_factory.")
        return setup_sdpo_teacher_regularization(
            teacher_regularization,
            student_module=student_module,
            reference_module=reference_module_factory(),
            teacher_update_rate=teacher_update_rate,
        )
    raise ValueError(f"unsupported SDPO teacher_regularization: {teacher_regularization!r}")


class SDPOTrustRegionTeacher(nn.Module):
    def __init__(self, reference: nn.Module, student: nn.Module, mix_coef: float) -> None:
        super().__init__()
        mix_coef = _validate_unit_interval(mix_coef, name="mix_coef")
        _validate_distinct_modules(reference, student, context="SDPO trust-region teacher")
        self.reference = reference
        self.student = student
        self.mix_coef = mix_coef
        self.reference.requires_grad_(False)
        self.reference.eval()

    def forward(self, *args, **kwargs):
        reference_out = self.reference(*args, **kwargs)
        student_out = self.student(*args, **kwargs)
        reference_logits = _extract_logits(reference_out)
        student_logits = _extract_logits(student_out)
        if reference_logits.shape != student_logits.shape:
            raise ValueError(
                "SDPO trust-region teacher logits shapes differ: "
                f"{tuple(reference_logits.shape)} != {tuple(student_logits.shape)}"
            )
        return SimpleNamespace(logits=torch.lerp(reference_logits, student_logits, self.mix_coef))


def _extract_logits(output) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    return output[0]


def sync_sdpo_teacher_from_student_(teacher: nn.Module, student: nn.Module) -> None:
    _validate_distinct_modules(teacher, student, context="SDPO EMA teacher")

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    if teacher_state.keys() != student_state.keys():
        teacher_only = sorted(teacher_state.keys() - student_state.keys())
        student_only = sorted(student_state.keys() - teacher_state.keys())
        raise ValueError(
            "SDPO EMA teacher and student state_dict keys differ "
            f"(teacher_only={teacher_only}, student_only={student_only})"
        )

    with torch.no_grad():
        for key, teacher_value in teacher_state.items():
            student_value = student_state[key]
            if teacher_value.shape != student_value.shape:
                raise ValueError(
                    "SDPO EMA teacher and student tensor shapes differ "
                    f"for {key!r}: {tuple(teacher_value.shape)} != {tuple(student_value.shape)}"
                )
            teacher_value.copy_(student_value.to(device=teacher_value.device, dtype=teacher_value.dtype))


def update_sdpo_ema_teacher_(teacher: nn.Module, student: nn.Module, update_rate: float) -> None:
    update_rate = _validate_unit_interval(update_rate, name="update_rate")
    _validate_distinct_modules(teacher, student, context="SDPO EMA teacher")
    if update_rate == 0.0:
        return

    teacher_params = list(teacher.named_parameters())
    student_params = list(student.named_parameters())
    if len(teacher_params) != len(student_params):
        raise ValueError(
            f"SDPO EMA teacher and student parameter counts differ: {len(teacher_params)} != {len(student_params)}"
        )

    with torch.no_grad():
        for idx, ((teacher_name, teacher_param), (student_name, student_param)) in enumerate(
            zip(teacher_params, student_params, strict=True)
        ):
            if teacher_name != student_name:
                raise ValueError(
                    "SDPO EMA teacher and student parameter names differ "
                    f"at index {idx}: {teacher_name!r} != {student_name!r}"
                )
            if teacher_param.shape != student_param.shape:
                raise ValueError(
                    "SDPO EMA teacher and student parameter shapes differ "
                    f"for {teacher_name!r}: {tuple(teacher_param.shape)} != {tuple(student_param.shape)}"
                )
            student_data = student_param.detach().to(device=teacher_param.device, dtype=teacher_param.dtype)
            teacher_param.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)
        _copy_sdpo_teacher_buffers_(teacher, student)


def _copy_sdpo_teacher_buffers_(teacher: nn.Module, student: nn.Module) -> None:
    teacher_buffers = list(teacher.named_buffers())
    student_buffers = list(student.named_buffers())
    if len(teacher_buffers) != len(student_buffers):
        raise ValueError(
            f"SDPO EMA teacher and student buffer counts differ: {len(teacher_buffers)} != {len(student_buffers)}"
        )
    for idx, ((teacher_name, teacher_buffer), (student_name, student_buffer)) in enumerate(
        zip(teacher_buffers, student_buffers, strict=True)
    ):
        if teacher_name != student_name:
            raise ValueError(
                f"SDPO EMA teacher and student buffer names differ at index {idx}: {teacher_name!r} != {student_name!r}"
            )
        if teacher_buffer.shape != student_buffer.shape:
            raise ValueError(
                "SDPO EMA teacher and student buffer shapes differ "
                f"for {teacher_name!r}: {tuple(teacher_buffer.shape)} != {tuple(student_buffer.shape)}"
            )
        teacher_buffer.copy_(student_buffer.to(device=teacher_buffer.device, dtype=teacher_buffer.dtype))
