import pickle

import torch

from prime_rl.inference.vllm.worker.nccl import receive_integer, receive_state_dict
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


class ReplayCommunicator:
    device = torch.device("cpu")

    def __init__(self, payloads: list[torch.Tensor]) -> None:
        self.payloads = iter(payloads)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        assert src == 0
        assert tensor.device == self.device
        tensor.copy_(next(self.payloads))


def test_nccl_receiver_ignores_meta_default_device() -> None:
    metadata = pickle.dumps({torch.float32: [("weight", (2,), 2)]})
    payloads = [
        torch.tensor([7], dtype=torch.long),
        torch.tensor([len(metadata)], dtype=torch.long),
        torch.tensor(list(metadata), dtype=torch.uint8),
        torch.tensor([1.0, 2.0]),
    ]
    communicator = ReplayCommunicator(payloads)

    with torch.device("meta"):
        integer = receive_integer(communicator)  # type: ignore[arg-type]
        state_dict = dict(receive_state_dict(communicator))  # type: ignore[arg-type]

    assert integer == 7
    torch.testing.assert_close(state_dict["weight"], torch.tensor([1.0, 2.0]))
