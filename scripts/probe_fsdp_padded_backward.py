"""Exercise mixed real and zero-loss FSDP2 backwards across data ranks."""

import argparse
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from prime_rl.trainer.rl.loss import attach_zero_loss_to_model_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--real-ranks", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if any(real_rank < 0 or real_rank >= world_size for real_rank in args.real_ranks):
        raise ValueError(f"real ranks must be in [0, {world_size}), got {args.real_ranks}")

    device = torch.device("cuda", local_rank)
    model = nn.Sequential(
        *(nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size), nn.SiLU()) for _ in range(args.num_layers)),
        nn.Linear(args.hidden_size, args.hidden_size),
    ).to(device=device, dtype=torch.bfloat16)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
    policy = CPUOffloadPolicy(pin_memory=True) if args.cpu_offload else None
    fsdp_kwargs = {
        "mesh": mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
    }
    if policy is not None:
        fsdp_kwargs["offload_policy"] = policy
    for layer in model:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    for step in range(args.num_steps):
        inputs = torch.randn(2, args.hidden_size, device=device, dtype=torch.bfloat16)
        output = model(inputs)
        real_loss = step < args.num_steps - 1 or rank in args.real_ranks
        loss = output.float().square().mean() if real_loss else output.new_zeros(())
        loss = attach_zero_loss_to_model_output(loss, {"logits": output})
        loss.backward()

    dist.barrier()
    if rank == 0:
        print(
            f"PASS world_size={world_size} steps={args.num_steps} "
            f"real_final_ranks={args.real_ranks} cpu_offload={args.cpu_offload}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
