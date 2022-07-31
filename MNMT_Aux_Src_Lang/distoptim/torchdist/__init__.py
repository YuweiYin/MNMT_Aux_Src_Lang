"""distoptim.torchdist package"""
import logging
import collections
import os
import os.path as op
import re
import json
import torch
import torch.distributed as dist

LOG = logging.getLogger(__name__)


def _get_philly_master_config_idx0():
    runtime_config_file = os.environ.get("PHILLY_RUNTIME_CONFIG")
    with open(runtime_config_file) as _f:
        runtime_config = json.load(_f)
        for name, config in runtime_config["containers"].items():
            if config["index"] == 0:
                master_config = config
    return master_config

def _get_philly_master_config_mpihost():
    runtime_config_file = os.environ.get("PHILLY_RUNTIME_CONFIG")
    with open(runtime_config_file) as _f:
        runtime_config = json.load(_f)
        path = runtime_config["scratchDirectory"]
        mpi_host_file = op.join(path, "mpi-hosts")
        with open(mpi_host_file, "r") as f:
            master_line = f.readline().strip()
            master_line_list = master_line.split(" ")
            master_name = master_line_list[0]
            master_name = master_name.replace("-", "_")
            master_config = runtime_config["containers"][master_name]
    return master_config

def _get_init_method_philly():
    # master_config = _get_philly_master_config_mpihost()
    master_config = _get_philly_master_config_idx0()
    master_ip = master_config["ip"]
    master_port = master_config["portRangeStart"] + 1
    return f"tcp://{master_ip}:{master_port}"

def _get_init_method_aml():
    # AZ_BATCHAI_MPI_MASTER_NODE or AZ_BATCHAI_JOB_MASTER_NODE_IP
    # "AZ_BATCH_MASTER_NODE": "10.0.0.6:6000" # only on multi node

    if "AZ_BATCH_MASTER_NODE" in os.environ:
        return "tcp://" + os.environ.get("AZ_BATCH_MASTER_NODE")
    
    master_ip = os.environ.get("AZ_BATCHAI_JOB_MASTER_NODE_IP")
    return f"tcp://{master_ip}:6000"

def _get_init_method():
    if "PHILLY_RUNTIME_CONFIG" in os.environ:
        return _get_init_method_philly()
    return _get_init_method_aml()


def rank():
    """Find OMPI world rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get("OMPI_COMM_WORLD_RANK") or 0)


def size():
    """Find OMPI world size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get("OMPI_COMM_WORLD_SIZE") or 1)


def local_rank():
    """Find OMPI local rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK") or 0)


def local_size():
    """Find OMPI local size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE") or 1)


def barrier():
    """Call dist.barrier()"""
    dist.barrier()


def broadcast(tensor, root_rank):
    """broadcast one tensor."""
    dist.broadcast(tensor, root_rank)


def local_init(backend):
    """Local init"""
    master_ip = "127.0.1.1"
    master_port = "50065"
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_ip}:{master_port}",
        world_size=size(),
        rank=rank(),
    )


def init(backend="nccl"):
    """Init torch distributed using TCP"""

    LOG.info("Initializing torch distributed")
    print("Initializing torch distributed")

    if dist.is_initialized():
        LOG.info("Torch distributed already initialized!")
        print("Torch distributed already initialized!")
        return

    init_method = _get_init_method()
    _world_size = size()
    _rank = rank()
    LOG.info(
        f"Init Method: {init_method}, World Size: {_world_size}, "
        f"Backend: {backend},  rank: {_rank}"
    )
    print(
        f"Init Method: {init_method}, World Size: {_world_size}, "
        f"Backend: {backend},  rank: {_rank}"
    )
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=_world_size,
        rank=_rank,
    )
    LOG.info("init_process_group done!")
    print("init_process_group done!")


def broadcast_parameters(params, root_rank):
    """Similar to horovod broadcast_parameters."""
    params = sorted(params.items())
    for _, _p in params:
        dist.broadcast(_p, root_rank)
    dist.barrier()
