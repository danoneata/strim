# run.py
import torch
import ignite.distributed as idist


def run(rank, config):
    print(f"Running basic DDP example on rank {rank}.")


def main():
    world_size = 4  # if this is 3 or more it hangs

    # some dummy config
    config = {}

    # run task
    # idist.spawn("nccl", run, args=(config,), nproc_per_node=world_size)

    # the same happens even in this case
    with idist.Parallel(backend="nccl", nproc_per_node=world_size) as parallel:
        parallel.run(run, config)


if __name__ == "__main__":
    main()
