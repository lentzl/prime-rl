import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from prime_rl.trainer.weights import load_state_dict, stream_sharded_state_dict


def test_stream_sharded_state_dict_writes_loadable_hf_index(tmp_path: Path) -> None:
    state_dict = {
        "first": torch.arange(8, dtype=torch.bfloat16),
        "second": torch.arange(8, 16, dtype=torch.bfloat16),
        "third": torch.arange(16, 24, dtype=torch.bfloat16),
    }

    stream_sharded_state_dict(
        state_dict,
        tmp_path,
        is_master=True,
        transform=lambda shard: {f"model.{key}": value for key, value in shard.items()},
        max_shard_size=16,
    )

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 48
    assert index["weight_map"] == {
        "model.first": "model-00001-of-00003.safetensors",
        "model.second": "model-00002-of-00003.safetensors",
        "model.third": "model-00003-of-00003.safetensors",
    }
    loaded = load_state_dict(tmp_path)
    assert set(loaded) == {"model.first", "model.second", "model.third"}
    torch.testing.assert_close(loaded["model.first"], state_dict["first"])
    torch.testing.assert_close(loaded["model.second"], state_dict["second"])
    torch.testing.assert_close(loaded["model.third"], state_dict["third"])


def _stream_dtensor_worker(rank: int, world_size: int, rendezvous: str, output_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        expected = torch.arange(48, dtype=torch.bfloat16).reshape(12, 4)
        state_dict = {
            "first": distribute_tensor(expected[:8], mesh, [Shard(0)]),
            "second": distribute_tensor(expected[8:], mesh, [Shard(0)]),
        }
        stream_sharded_state_dict(
            state_dict,
            Path(output_dir),
            is_master=rank == 0,
            transform=lambda shard: shard,
            max_shard_size=32,
        )
        dist.barrier()
        if rank == 0:
            loaded = load_state_dict(Path(output_dir))
            torch.testing.assert_close(loaded["first"], expected[:8])
            torch.testing.assert_close(loaded["second"], expected[8:])
    finally:
        dist.destroy_process_group()


def test_stream_sharded_state_dict_gathers_dtensors(tmp_path: Path) -> None:
    mp.spawn(
        _stream_dtensor_worker,
        args=(2, str(tmp_path / "rendezvous"), str(tmp_path / "weights")),
        nprocs=2,
        join=True,
    )
