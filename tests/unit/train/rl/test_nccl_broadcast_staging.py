import torch

from prime_rl.trainer.rl.broadcast.nccl import broadcast_integer, broadcast_state_dict


class RecordingCommunicator:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.broadcasts: list[torch.Tensor] = []

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        assert src == 0
        assert tensor.device == self.device
        self.broadcasts.append(tensor.clone())


def test_nccl_broadcast_stages_cpu_state_on_communicator_device() -> None:
    communicator = RecordingCommunicator()

    broadcast_integer(3, communicator)  # type: ignore[arg-type]
    broadcast_state_dict(
        {
            "float_weight": torch.tensor([1.0, 2.0]),
            "integer_weight": torch.tensor([3, 4]),
        },
        communicator,  # type: ignore[arg-type]
    )

    assert len(communicator.broadcasts) == 5
    torch.testing.assert_close(communicator.broadcasts[0], torch.tensor([3]))
    assert all(tensor.device == communicator.device for tensor in communicator.broadcasts)
