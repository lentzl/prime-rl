import json
import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"expected two ranks, found {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    value = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(value)
    if value.item() != 3.0:
        raise RuntimeError(f"unexpected NCCL all-reduce result: {value.item()}")

    generator = torch.Generator(device="cuda").manual_seed(3806011 + rank)
    left = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16, generator=generator)
    right = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16, generator=generator)
    product = left @ right
    torch.cuda.synchronize()
    if not torch.isfinite(product).all().item():
        raise RuntimeError("BF16 matrix product contains a non-finite value")

    print(
        json.dumps(
            {
                "rank": rank,
                "device": torch.cuda.get_device_name(local_rank),
                "capability": torch.cuda.get_device_capability(local_rank),
                "nccl_all_reduce": value.item(),
                "bf16_matmul_finite": True,
                "bf16_matmul_checksum": float(product.float().sum().item()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
