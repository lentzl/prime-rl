from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor


def _validate_update_rate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"update_rate must be a finite numeric value in [0, 1], got {value!r}")
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"update_rate must be in [0, 1], got {value}")
    return value


def _paired_named_tensors(
    teacher_tensors: list[tuple[str, torch.Tensor]],
    student_tensors: list[tuple[str, torch.Tensor]],
    *,
    kind: str,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    if len(teacher_tensors) != len(student_tensors):
        raise ValueError(f"SDPO EMA teacher and student {kind} counts differ")
    pairs = []
    for (teacher_name, teacher_value), (student_name, student_value) in zip(
        teacher_tensors, student_tensors, strict=True
    ):
        if teacher_name != student_name:
            raise ValueError(f"SDPO EMA teacher and student {kind} names differ")
        if teacher_value.shape != student_value.shape:
            raise ValueError(f"SDPO EMA teacher and student shapes differ for {teacher_name!r}")
        pairs.append((teacher_name, teacher_value, student_value))
    return pairs


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def _paired_local_tensors(
    teacher_tensors: list[tuple[str, torch.Tensor]],
    student_tensors: list[tuple[str, torch.Tensor]],
    *,
    kind: str,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    pairs = []
    for name, teacher_value, student_value in _paired_named_tensors(
        teacher_tensors,
        student_tensors,
        kind=kind,
    ):
        teacher_local = _local_tensor(teacher_value)
        student_local = _local_tensor(student_value)
        if teacher_local.shape != student_local.shape:
            raise ValueError(f"SDPO EMA teacher and student local shard shapes differ for {name!r}")
        pairs.append((name, teacher_local, student_local))
    return pairs


class SDPOEMATeacher(Stateful):
    """Checkpointable EMA teacher bound to the trainable policy."""

    def __init__(self, teacher: nn.Module, student: nn.Module, update_rate: float):
        if teacher is student:
            raise ValueError("SDPO EMA teacher must be a separate model")
        self.teacher = teacher
        self.student = student
        self.update_rate = _validate_update_rate(update_rate)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def state_dict(self) -> dict[str, Any]:
        return {"update_rate": self.update_rate, "teacher": self.teacher.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.update_rate = _validate_update_rate(state_dict["update_rate"])
        self.teacher.load_state_dict(state_dict["teacher"])

    def sync_from_student(self) -> None:
        with torch.no_grad():
            for _, teacher_value, student_value in _paired_local_tensors(
                list(self.teacher.named_parameters()),
                list(self.student.named_parameters()),
                kind="parameter",
            ) + _paired_local_tensors(
                list(self.teacher.named_buffers()),
                list(self.student.named_buffers()),
                kind="buffer",
            ):
                teacher_value.copy_(student_value.to(device=teacher_value.device, dtype=teacher_value.dtype))

    def step(self) -> None:
        with torch.no_grad():
            for _, teacher_value, student_value in _paired_local_tensors(
                list(self.teacher.named_parameters()),
                list(self.student.named_parameters()),
                kind="parameter",
            ):
                student_value = student_value.to(device=teacher_value.device, dtype=teacher_value.dtype)
                teacher_value.lerp_(student_value, self.update_rate)
