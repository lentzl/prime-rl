import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

from prime_rl.trainer.rl.sdpo_teacher import SDPOEMATeacher


class ModuleWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.register_buffer("scale", torch.zeros(1))


def test_sdpo_ema_teacher_syncs_state_and_only_averages_parameters():
    teacher = ModuleWithBuffer()
    student = ModuleWithBuffer()
    with torch.no_grad():
        teacher.linear.weight.zero_()
        teacher.linear.bias.zero_()
        teacher.scale.zero_()
        student.linear.weight.copy_(torch.tensor([[2.0, 4.0]]))
        student.linear.bias.copy_(torch.tensor([6.0]))
        student.scale.copy_(torch.tensor([8.0]))

    ema = SDPOEMATeacher(teacher, student, update_rate=0.25)
    ema.sync_from_student()
    torch.testing.assert_close(teacher.linear.weight, student.linear.weight)
    torch.testing.assert_close(teacher.scale, student.scale)

    with torch.no_grad():
        student.linear.weight.copy_(torch.tensor([[6.0, 8.0]]))
        student.linear.bias.copy_(torch.tensor([10.0]))
        student.scale.copy_(torch.tensor([12.0]))
    ema.step()

    torch.testing.assert_close(teacher.linear.weight, torch.tensor([[3.0, 5.0]]))
    torch.testing.assert_close(teacher.linear.bias, torch.tensor([7.0]))
    torch.testing.assert_close(teacher.scale, torch.tensor([8.0]))


def test_sdpo_ema_teacher_state_round_trips():
    source = SDPOEMATeacher(ModuleWithBuffer(), ModuleWithBuffer(), update_rate=0.01)
    with torch.no_grad():
        source.teacher.linear.weight.fill_(3.0)
        source.teacher.scale.fill_(5.0)

    restored = SDPOEMATeacher(ModuleWithBuffer(), ModuleWithBuffer(), update_rate=0.5)
    restored.load_state_dict(source.state_dict())

    assert restored.update_rate == 0.01
    torch.testing.assert_close(restored.teacher.linear.weight, source.teacher.linear.weight)
    torch.testing.assert_close(restored.teacher.scale, source.teacher.scale)


@pytest.mark.parametrize("update_rate", [-0.1, 1.1, float("nan"), True])
def test_sdpo_ema_teacher_rejects_invalid_update_rate(update_rate):
    with pytest.raises(ValueError, match="update_rate"):
        SDPOEMATeacher(nn.Linear(1, 1), nn.Linear(1, 1), update_rate=update_rate)


def test_sdpo_ema_teacher_freezes_teacher():
    teacher = nn.Linear(1, 1)
    ema = SDPOEMATeacher(teacher, nn.Linear(1, 1), update_rate=0.01)

    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert ema.teacher is teacher


def test_sdpo_ema_teacher_updates_from_dtensor_local_shard(tmp_path):
    store_path = tmp_path / "ema-dtensor-store"
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=0, world_size=1)
    try:
        mesh = DeviceMesh("cpu", [0])
        teacher = nn.Linear(2, 1, bias=False)
        student = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            teacher.weight.zero_()
            student.weight.copy_(torch.tensor([[4.0, 8.0]]))
        student.weight = nn.Parameter(distribute_tensor(student.weight.detach(), mesh, [Shard(0)]))

        ema = SDPOEMATeacher(teacher, student, update_rate=0.25)
        ema.step()

        torch.testing.assert_close(teacher.weight, torch.tensor([[1.0, 2.0]]))
    finally:
        dist.destroy_process_group()
