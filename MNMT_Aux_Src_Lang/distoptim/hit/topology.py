"""distoptim.hit package"""
import logging
import torch.distributed as dist
import numpy as np


LOG = logging.getLogger(__name__)


class SyncNode(object):
    """Sync Node, represents one HiT layer"""
    def __init__(self, master_rank, my_rank, group_idx, level, sync_factor):
        self.master_rank = master_rank
        self.my_rank = my_rank
        self.group_idx = group_idx
        self.level = level
        self.sync_factor = sync_factor
        self.optim_type = ""
        self.broadcast_group = None
        self.broadcast_members = None
        self.reduce_group = None
        self.reduce_members = None
        self.next_ = None
        self.prev_ = None
        self.model_state_dict = None
        self.optimizer = None
        self.lrs = None
        self.lrscheduler = None
        self.sync_time_threshold = 10000
        self.sync_offset = 0
        self.last_sync_step = 0

    def __repr__(self):
        repr_str = (
            f"<SyncNode at level {self.level},"
            f" master {self.master_rank}, reduce: {self.reduce_members}"
            f" bcast size {self.broadcast_members},"
            f" sync factor {self.sync_factor}>")
        return repr_str


def create_sync_nodes(rank, topology, sync_factors, verbose=False):
    """
    topology: 8 32 (first level sync in 8 gpus, second level sync in 32 gpus)
    sync_factors: 50 500 (first level do sync every 50 steps, second level do
                  sync every 500 steps)
    """
    if dist.get_world_size() == 1:
        return []

    world_size = dist.get_world_size()
    sync_nodes = []

    for level in range(len(topology)):
        if verbose:
            LOG.info(f"Level {level}")
            print(f"Level {level}")
        prev_topo = topology[level - 1] if level > 0 else 1
        all_members = np.arange(world_size).reshape(-1, topology[level])
        for group_idx, row in enumerate(all_members):
            reduce_group_ranks = row[::prev_topo].tolist()
            reduce_gp = dist.new_group(ranks=reduce_group_ranks)
            if verbose:
                LOG.info(f"\tReduce group {reduce_group_ranks}")
                print(f"\tReduce group {reduce_group_ranks}")

            bcast_group_ranks = row.tolist()
            bcast_gp = dist.new_group(ranks=bcast_group_ranks)
            if verbose:
                LOG.info(f"\tBcast group {bcast_group_ranks}")
                print(f"\tBcast group {bcast_group_ranks}")

            if rank in row:
                sync_node = SyncNode(row[0], rank, group_idx, level, sync_factors[level])
                sync_node.broadcast_group = bcast_gp
                sync_node.reduce_group = reduce_gp
                sync_node.broadcast_members = bcast_group_ranks
                sync_node.reduce_members = reduce_group_ranks
        sync_nodes.append(sync_node)
    sync_nodes = [None] + sync_nodes + [None]
    for i in range(1, len(sync_nodes) - 1):
        sync_nodes[i].next_ = sync_nodes[i + 1]
        sync_nodes[i].prev_ = sync_nodes[i - 1]
    if verbose:
        LOG.info(sync_nodes[1:-1])
        print(sync_nodes[1:-1])
    return sync_nodes[1:-1]
