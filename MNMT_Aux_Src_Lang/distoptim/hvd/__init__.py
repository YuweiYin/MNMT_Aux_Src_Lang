"""distoptim.hvd package"""
import logging
import torch
import horovod.torch as hvd


LOG = logging.getLogger(__name__)


def init():
    """Init horovod."""
    LOG.info("Using Horovod for distributed training")
    hvd.init()


def size():
    """Get the world size."""
    return hvd.size()


def rank():
    """Get my rank."""
    return hvd.rank()


def local_size():
    """Get local size."""
    return hvd.local_size()


def local_rank():
    """Get local rank."""
    return hvd.local_rank()


def barrier():
    """Similar to torch distributed Barrier."""
    hvd.allreduce(torch.tensor(0), name="barrier")


def broadcast(tensor, root_rank):
    """broadcast."""
    hvd.broadcast(tensor, root_rank, name="broadcast_tensor")


def broadcast_parameters(params, root_rank):
    """Call horovod broadcast_parameters."""
    hvd.broadcast_parameters(params, root_rank)


def create_distributed_optimizer(model, optimizer, config):
    """Call horovod DistributedOptimizer."""
    compression = (
        hvd.Compression.fp16
        if "compression" in config and config["compression"] == "fp16"
        else hvd.Compression.none
    )
    backward_passes_per_step = (
        int(config["backward_passes_per_step"])
        if "backward_passes_per_step" in config
        else 1
    )
    return hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=backward_passes_per_step,
    )
