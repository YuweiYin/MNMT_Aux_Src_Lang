"""distoptim.hit package"""
import logging
import torch
import apex
from apex.parallel.distributed import flatten, unflatten
from apex.multi_tensor_apply import multi_tensor_applier
import torch.distributed as dist


LOG = logging.getLogger(__name__)


class _HiT_APEX_DDP(apex.parallel.DistributedDataParallel):
    """ Wrap input model with torch DistributedDataParallel
        And redirect attributes
    """
    def __init__(
            self,
            module,
            delay_allreduce=False,
            process_group=None,
    ):

        LOG.info("Init _HiT_APEX_DDP")

        super(_HiT_APEX_DDP, self).__init__(
            module,
            delay_allreduce=delay_allreduce,
        )

        self.process_group = process_group
        self.world_size = float(dist.get_world_size(group=process_group))

    # def __getattr__(self, attr):
    #     # may not be necessary for this class
    #     # circular calls from __init__
    #     if attr == "module":
    #         return super(_HiT_APEX_DDP, self).__getattr__(attr)

    #     # redirect
    #     return getattr(self.module, attr)

    def load_state_dict(self, state_dict, strict=True):
        load_state_dict_on_module = True
        for key in state_dict.keys():
            if 'module.' in key:
                load_state_dict_on_module = False
                break
        if load_state_dict_on_module:
            self.module.load_state_dict(state_dict, strict=strict)
        else:
            super(_HiT_APEX_DDP, self).load_state_dict(state_dict, strict=strict)

    # an older version from apex
    def allreduce_bucket(self, bucket):
        tensor = flatten(bucket)

        tensor_to_allreduce = tensor 

        if self.allreduce_always_fp32:
            tensor_to_allreduce = tensor.float() 

        if self.gradient_predivide_factor != 1.0:
            tensor_to_allreduce.mul_(1./self.gradient_predivide_factor)

        dist.all_reduce(tensor_to_allreduce, group=self.process_group)

        if self.gradient_average:
            if self.gradient_predivide_factor != self.world_size:
                tensor_to_allreduce.mul_(self.gradient_predivide_factor/self.world_size)

        if self.allreduce_always_fp32 and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)
 
        return tensor
    

    # latest apex master version
    def allreduce_bucket(self, bucket, bucket_idx, force_default_stream):
        '''Override the default one to use sub group all reduce'''
        tensor = flatten(bucket)

        if force_default_stream:
            bucket_stream = self.main_stream
        else:
            bucket_stream = self._stream_this_bucket(bucket_idx)
            bucket_event = self._event_this_bucket(bucket_idx)
            torch.cuda.current_stream().record_event(bucket_event)
            bucket_stream.wait_event(bucket_event)

        with torch.cuda.stream(bucket_stream):
            # self.main_stream.wait_stream(torch.cuda.current_stream())
            # torch.cuda.synchronize()

            tensor_to_allreduce = tensor

            if self.allreduce_always_fp32:
                tensor_to_allreduce = tensor.float()

            if self.gradient_predivide_factor != 1.0:
                tensor_to_allreduce.mul_(1./self.gradient_predivide_factor)

            assert(not self.allreduce_different_streams)
            dist.all_reduce(tensor_to_allreduce, group=self.process_group)

            if self.gradient_average:
                tensor_to_allreduce.mul_(self.gradient_predivide_factor/self.world_size)

            if self.allreduce_always_fp32 and tensor is not tensor_to_allreduce:
                tensor.copy_(tensor_to_allreduce)

            if not self.retain_allreduce_buffers:
                if multi_tensor_applier.available:
                    multi_tensor_applier(
                        self.multi_tensor_scale,
                        self._overflow_buf,
                        [unflatten(tensor, bucket), bucket],
                        1.0)
                else:
                    for buf, synced in zip(bucket, unflatten(tensor, bucket)):
                        buf.copy_(synced)


            # I think we actually do need this here.  After allreduce_bucket returns, tensor will
            # eventually go out of scope and die, at which point it could otherwise be freed for
            # further reuse by the main stream while the allreduce/div/unflatten are underway in bucket_stream.
            tensor.record_stream(bucket_stream)

        return tensor

def wrap_model_with_apex_ddp(
        module,
        delay_allreduce=False,
        process_group=None,
):
    """ Wrap input model with torch DistributedDataParallel
    """
    return _HiT_APEX_DDP(
        module,
        delay_allreduce=delay_allreduce,
        process_group=process_group,
    )
