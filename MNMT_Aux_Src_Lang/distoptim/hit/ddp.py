"""distoptim.hit package"""
import logging
import torch


LOG = logging.getLogger(__name__)


class _HiT_DDP(torch.nn.parallel.DistributedDataParallel):
    """ Wrap input model with torch DistributedDataParallel
        And redirect attributes
    """
    def __init__(
            self,
            module,
            device_ids=None,
            output_device=None,
            dim=0,
            broadcast_buffers=True,
            process_group=None,
            bucket_cap_mb=25,
            find_unused_parameters=False,
            check_reduction=False,
    ):

        LOG.info("Init _HiT_DDP")

        super(_HiT_DDP, self).__init__(
            module,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            process_group=process_group,
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
            check_reduction=check_reduction,
        )

    def __getattr__(self, attr):
        # circular calls from __init__
        if attr == "module":
            return super(_HiT_DDP, self).__getattr__(attr)

        # redirect
        return getattr(self.module, attr)

    def load_state_dict(self, state_dict, strict=True):
        load_state_dict_on_module = True
        for key in state_dict.keys():
            if 'module.' in key:
                load_state_dict_on_module = False
                break
        if load_state_dict_on_module:
            self.module.load_state_dict(state_dict, strict=strict)
        else:
            super(_HiT_DDP, self).load_state_dict(state_dict, strict=strict)


def wrap_model_with_ddp(
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        check_reduction=False,
):
    """ Wrap input model with torch DistributedDataParallel
    """

    return _HiT_DDP(
        module,
        device_ids=device_ids,
        output_device=output_device,
        dim=dim,
        broadcast_buffers=broadcast_buffers,
        process_group=process_group,
        bucket_cap_mb=bucket_cap_mb,
        find_unused_parameters=find_unused_parameters,
        check_reduction=check_reduction,
    )
