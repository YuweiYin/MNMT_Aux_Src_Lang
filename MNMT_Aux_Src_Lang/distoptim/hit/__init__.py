"""distoptim.hit package"""
import logging
import torch
import torch.distributed as dist
from .hitoptimizer import HiTDistributedOptimizer
from .topology import create_sync_nodes

LOG = logging.getLogger(__name__)

__version__ = "0.1.1"


def _validate_config(config):
    """Validate configuration."""

    # validate total_num_workers, should match world size
    world_size = dist.get_world_size()
    world_size_config = config["total_num_workers"]
    if world_size_config != world_size:
        raise ValueError(
            f"ERROR: Invalid total_num_gpus - {world_size_config} does not"
            f" match runtime world_size {world_size}"
        )

    # validate HiT configurations
    hit_config = config["hit"]
    dist_opt_config = hit_config["distributed_optimizers"]

    # at least 1 layer
    total_layers = len(dist_opt_config)
    if total_layers < 1:
        raise ValueError(
            f"ERROR: distributed_optimizers need at least one layer"
        )

    last_sync_factor = 0
    for i, cfg in enumerate(dist_opt_config):
        # sync_factor should be ascending sequence of 1,n,m...
        if i == 0:
            if cfg["sync_factor"] != 1:
                LOG.info("WARNING: forcing sync_factor to 1 for layer 0")
                cfg["sync_factor"] = 1
        else:
            if cfg["sync_factor"] < last_sync_factor:
                raise ValueError(
                    f"ERROR: Invalid sync_factor for layer {i}, should be "
                    "bigger than the lower layers"
                )
        last_sync_factor = cfg["sync_factor"]

        # sync_partition should be descending sequence of x,..y,1
        if i == total_layers - 1:
            if cfg["sync_partition"] != 1:
                LOG.info(
                    "WARNING: forcing sync_partition of top layer {i} to 1"
                )
                cfg["sync_partition"] = 1
        else:
            if world_size % cfg["sync_partition"] != 0:
                raise ValueError(
                    f"ERROR: Invalid sync_partition for layer {i}, "
                    f"should be divisible by {world_size}"
                )


def create_distributed_optimizer(model, optimizer, config):
    """Create HiTDistributedOptimizer."""

    LOG.info(f"HiT version {__version__}")
    print(f"HiT version {__version__}")


    torch.cuda.empty_cache()

    _validate_config(config)

    hit_config = config["hit"]
    hit_config["version"] = __version__
    dist_opt_config = hit_config["distributed_optimizers"]

    world_size = dist.get_world_size()
    sync_factors = [c["sync_factor"] for c in dist_opt_config]
    topology = [world_size // c["sync_partition"] for c in dist_opt_config]

    sync_nodes = create_sync_nodes(
        dist.get_rank(), topology, sync_factors, verbose=True
    )

    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        dict(HiTDistributedOptimizer.__dict__),
    )

    return cls(
        sync_nodes,
        model,
        optimizer,
        dist_opt_config,
        hit_config,
    )
