"""Probe a large NCCL reduce-scatter using the current torchrun world."""

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-numel", type=int, default=383_418_612)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=args.timeout_seconds),
        device_id=torch.device("cuda", local_rank),
    )

    world_size = dist.get_world_size()
    if args.input_numel % world_size:
        raise ValueError("--input-numel must be divisible by the world size")

    output_numel = args.input_numel // world_size
    input_tensor = torch.ones(args.input_numel, dtype=torch.bfloat16, device="cuda")
    output_tensor = torch.empty(output_numel, dtype=torch.bfloat16, device="cuda")

    for iteration in range(args.iterations):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.monotonic()
        dist.reduce_scatter_tensor(output_tensor, input_tensor)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        timings = [None] * world_size if dist.get_rank() == 0 else None
        dist.gather_object(elapsed, timings, dst=0)
        if timings is not None:
            print(
                f"iteration={iteration + 1} input_numel={args.input_numel} "
                f"min_seconds={min(timings):.3f} max_seconds={max(timings):.3f}",
                flush=True,
            )

    expected = torch.tensor(float(world_size), dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(output_tensor[0], expected)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
