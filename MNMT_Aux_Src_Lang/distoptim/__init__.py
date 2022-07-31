"""distoptim package"""
import logging
import distoptim.hit as dohit
import distoptim.torchdist as tdist


_DO_TYPE = "horovod"
_DO_CONFIG = {}


def init(config):
    """Init distributed optimizer according to config."""
    global _DO_CONFIG
    global _DO_TYPE
    _DO_CONFIG = config
    _DO_TYPE = config["type"] if "type" in config else "horovod"
    _backend = config["backend"] if "backend" in config else "nccl"
    _is_local = config["is_local"] if "is_local" in config else False

    if _DO_TYPE == "horovod":
        import distoptim.hvd as dohvd
        dohvd.init()
    elif _DO_TYPE == "hit" or _DO_TYPE == "torchdist":
        if _is_local:
            tdist.local_init(_backend)
        else:
            tdist.init(_backend)
    else:
        raise ValueError("ERROR: type should be one of horovod or hit or torchdist!")


def size():
    """Get the world size."""
    if _DO_TYPE == "horovod":
        import distoptim.hvd as dohvd
        return dohvd.size()
    else:
        return tdist.size()


def rank():
    """Get my rank."""
    if _DO_TYPE == "horovod":
        import distoptim.hvd as dohvd
        return dohvd.rank()
    else:
        return tdist.rank()


def local_size():
    """Get local size."""
    if _DO_TYPE == "horovod":
        import distoptim.hvd as dohvd
        return dohvd.local_size()
    else:
        return tdist.local_size()


def local_rank():
    """Get local rank."""
    if _DO_TYPE == "horovod":
        import distoptim.hvd as dohvd
        return dohvd.local_rank()
    else:
        return tdist.local_rank()


def barrier():
    """Barrier."""
    if _DO_TYPE == "horovod":
        import distoptim.hvd as dohvd
        dohvd.barrier()
    else:
        tdist.barrier()


def broadcast(tensor, root_rank):
    """broadcast."""
    if _DO_TYPE == "horovod":
        import distoptim.hvd as dohvd
        dohvd.broadcast(tensor, root_rank)
    else:
        tdist.broadcast(tensor, root_rank)


def broadcast_parameters(params, root_rank):
    """Similar to horovod broadcast_parameters."""
    if _DO_TYPE == "horovod":
        import distoptim.hvd as dohvd
        dohvd.broadcast_parameters(params, root_rank)
    else:
        tdist.broadcast_parameters(params, root_rank)


def create_distributed_optimizer(model, optimizer):
    """Init distributed optimizer according to config."""
    if _DO_TYPE == "horovod":
        import distoptim.hvd as dohvd
        return dohvd.create_distributed_optimizer(
            model, optimizer, _DO_CONFIG)
    elif _DO_TYPE == "hit":
        return dohit.create_distributed_optimizer(
            model, optimizer, _DO_CONFIG)
    else:
        raise ValueError("ERROR: create_distributed_optimizer is only valid for type horovod or hit!")
