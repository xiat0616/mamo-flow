import os
import sys
import torch
import torch.distributed as dist

def log(msg: str):
    print(msg, flush=True)

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    x = torch.ones(1, device=f"cuda:{local_rank}") * (rank + 1)
    dist.all_reduce(x)
    torch.cuda.synchronize()

    log(f"rank={rank} local_rank={local_rank} world_size={world_size} x={x.item()}")

    dist.barrier()
    log(f"rank={rank} passed barrier")

    dist.destroy_process_group()
    log(f"rank={rank} done")

if __name__ == "__main__":
    main()