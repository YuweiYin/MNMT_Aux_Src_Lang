"""distoptim.hit package"""
import logging
import math
import copy
import time
import torch
import torch.distributed as dist
from .lrschedular import CustomStepLR
from .lars import LarsSGD, LarsSGDV1
from .lamb import LambSGD
from distoptim.torchdist import local_rank, rank

#import tensorboard_logger as tbl

LOG = logging.getLogger(__name__)


def get_config(config, key, default):
    return config[key] if key in config else default

def init_logging(hit_config):
    logging_stream = get_config(hit_config, "logging_stream", None)
    logging_level = get_config(hit_config, "logging_level", None)

    if logging_stream == "stdout":
        logging.basicConfig()

    log_level = logging.INFO
    rank0_only = True
    if logging_level == "verbose":
        log_level = logging.DEBUG
        rank0_only = False
    elif logging_level == "debug":
        log_level = logging.DEBUG

    effective_level = LOG.getEffectiveLevel()
    if ((effective_level == 0 or effective_level > 20)
            and (not rank0_only or rank() == 0)):
        LOG.setLevel(log_level)

def print_cuda_stats(msg=None):
    if msg:
        LOG.debug(msg)
    LOG.debug("torch.cuda.memory_allocated(): {}".format(torch.cuda.memory_allocated()))
    LOG.debug("torch.cuda.memory_cached(): {}".format(torch.cuda.memory_cached()))

def reset_optimizer(optimizer):
    """reset optimizer states"""
    optimizer_states = optimizer.state_dict()["state"]
    for param_id in optimizer_states:
        for state in optimizer_states[param_id]:
            if isinstance(optimizer_states[param_id][state], torch.Tensor):
                optimizer_states[param_id][state].zero_()


class HiTDistributedOptimizer(torch.optim.Optimizer):
    """HiTDistributedOptimizer"""

    def __init__(
            self,
            sync_nodes,
            model,
            optimizer,
            dist_config,
            hit_config
    ):

        init_logging(hit_config)

        LOG.info("HiT - creating HitDistributedOptimizer")

        super(self.__class__, self).__init__(optimizer.param_groups)

        # self.tb_logger = tbl.Logger(LOG)
        self.sync_nodes = sync_nodes
        self.model = model
        self.optimizer = optimizer
        self.hit_config = hit_config
        self.sync_local_optimizer = get_config(
            hit_config, "sync_local_optimizer", False
        )
        self.reset_after_sync = get_config(
            hit_config, "reset_after_sync", False
        )
        self.version = get_config(hit_config, "version", "")
        self._num_steps = 0
        self._model_updated = False
        self._is_inverse_pyramid = get_config(
            hit_config, "inverse_pyramid", True
        )
        self._fp16 = False
        self._init_fp16()

        LOG.debug(self.hit_config)

        print_cuda_stats("before init layers")
        for i, (layer_config, sync_node) in enumerate(zip(
                dist_config, self.sync_nodes)):
            if i == 0:
                self._init_layer_0(layer_config, sync_node)
            else:
                self._init_sync_node(i, layer_config, sync_node)
            print_cuda_stats(f"after init layer {i}")

    def state_dict(self):

        new_state_dict = super(self.__class__, self).state_dict()

        hit_state = self._get_hit_state()
        new_state_dict['hit_state'] = hit_state
        return new_state_dict

    def load_state_dict(self, state_dict):

        hit_state = None
        if 'hit_state' in state_dict:
            hit_state = state_dict['hit_state']
            del state_dict['hit_state']
        else:
            LOG.info("hit_state not found in state_dict")

        # first load on base optimizer
        super(self.__class__, self).load_state_dict(state_dict)

        if hit_state:
            LOG.info("load HiT states")
            self._load_hit_state(hit_state)

    def _get_hit_state(self):

        layer_states = []
        for idx, sync_node in enumerate(self.sync_nodes):
            layerx_state = {
                'layer_id': idx,
                'optim_type': sync_node.optim_type,
                'last_sync_step': sync_node.last_sync_step,
            }
            if sync_node.model_state_dict:
                layerx_state['model_state_dict'] = sync_node.model_state_dict
            if sync_node.optimizer:
                layerx_state['optim_state_dict'] = sync_node.optimizer.state_dict()
            layer_states.append(layerx_state)

        hit_state = {
            'version': self.version,
            'num_steps': self._num_steps,
            'layer_states': layer_states,
        }
        return hit_state

    def _load_hit_state(self, hit_state):
        hit_state_version = hit_state['version']
        LOG.info(f"restoring hit_state version {hit_state_version}")

        self._num_steps = hit_state['num_steps']
        LOG.info(f"  restoring _num_steps {self._num_steps}")

        LOG.info("  restoring layers")
        layer_states = hit_state['layer_states']
        if len(layer_states) != len(self.sync_nodes):
            raise RuntimeError("Invalid checkpoint, layers mismatch checkpoint {0} - us {1}".format(len(layer_states), len(self.sync_nodes)))

        for layerx_state in layer_states:
            layer_id = layerx_state['layer_id']
            LOG.info(f"    restoring layer {layer_id}")

            assert(layer_id < len(self.sync_nodes))
            sync_node = self.sync_nodes[layer_id]

            assert(layerx_state['optim_type'] == sync_node.optim_type)
            sync_node.last_sync_step = layerx_state['last_sync_step']

            if sync_node.model_state_dict:
                LOG.info(f"        restoring model state_dict")
                for name in sync_node.model_state_dict:
                    assert(name in layerx_state['model_state_dict'])
                    sync_node.model_state_dict[name].copy_(layerx_state['model_state_dict'][name])

            if sync_node.optimizer:
                LOG.info(f"        restoring optimizer state_dict")
                sync_node.optimizer.load_state_dict(layerx_state['optim_state_dict'])

            if sync_node.lrscheduler:
                LOG.info(f"        restoring lrscheduler")
                sync_node.lrscheduler.step(self._num_steps)

    def _init_fp16(self):
        optim_fp32 = False
        for group in self.param_groups:
            for p in group['params']:
                if p.data.dtype == torch.float or p.data.dtype == torch.double:
                    optim_fp32 = True
                    break
        for name, param in self.model.named_parameters():
            if param.data.dtype == torch.float16 and optim_fp32:
                LOG.info("setting _fp16 as model is fp16 but optimizer params is fp32")
                self._fp16 = True
            break

    def _init_layer_0(self, config, sync_node):
        optim_type = config["type"]
        sync_node.optim_type = optim_type.lower()

        if sync_node.optim_type == "ddp":
            from .ddp import wrap_model_with_ddp
            self.model = wrap_model_with_ddp(
                self.model,
                process_group=sync_node.broadcast_group,
                device_ids=[local_rank()],
                output_device=local_rank(),
                broadcast_buffers=False
            )
        elif sync_node.optim_type == "apex_ddp":
            from .apex_ddp import wrap_model_with_apex_ddp
            delay_allreduce = get_config(config, "delay_allreduce", True)
            self.model = wrap_model_with_apex_ddp(
                self.model,
                delay_allreduce=delay_allreduce,
                process_group=sync_node.broadcast_group,
            )
        else:
            assert(optim_type.lower() == "avg")

    def _init_sync_node(self, id, config, sync_node):
        optim_type = config["type"]
        sync_node.optim_type = optim_type.lower()
        sync_node.sync_time_threshold = get_config(
            config, "sync_time_threshold", 10000
        )
        sync_node.sync_offset = get_config(config, "sync_offset", 0)
        LOG.info(f"_init_sync_node, layer {id}, {optim_type}, {sync_node.sync_offset} ")

        if optim_type.lower() == "avg":
            return

        if (not self._is_inverse_pyramid and
                sync_node.my_rank not in sync_node.reduce_members):
            return
        
        lr = get_config(config, "learning_rate", 1.0)
        momentum = get_config(config, "momentum", 0.0)
        weight_decay = get_config(config, "weight_decay", 0.0)
        betas = (
            tuple(config["betas"])
            if "betas" in config
            else (0.9, 0.999)
        )
        eps = get_config(config, "eps", 1e-6)
        adam = get_config(config, "adam", False)
        amsgrad = get_config(config, "amsgrad", False)

        sync_node.model_state_dict = copy.deepcopy(self.model.state_dict())

        # init grad buffer
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                sync_node.model_state_dict[name].grad = torch.zeros_like(
                    param, requires_grad=False
                )

        if optim_type == "ma":
            lr = 1.0
            momentum = 0.0
            sync_node.optimizer = torch.optim.SGD(
                sync_node.model_state_dict.values(),
                lr=lr,
                momentum=momentum,
                dampening=momentum,
                weight_decay=weight_decay,
            )
        elif optim_type == "sgd":
            sync_node.optimizer = torch.optim.SGD(
                sync_node.model_state_dict.values(),
                lr=lr,
                momentum=momentum,
                dampening=momentum,
                weight_decay=weight_decay,
            )
        elif optim_type == "adam":
            sync_node.optimizer = torch.optim.Adam(
                sync_node.model_state_dict.values(),
                lr=lr,
                betas=betas,
            )
        elif optim_type == "larsv1":
            sync_node.optimizer = LarsSGDV1(
                sync_node.model_state_dict.values(),
                lr=lr,
                momentum=momentum,
                dampening=momentum,
                weight_decay=weight_decay,
            )
        elif optim_type == "lars":
            sync_node.optimizer = LarsSGD(
                sync_node.model_state_dict.values(),
                lr=lr,
                momentum=momentum,
                dampening=momentum,
                weight_decay=weight_decay,
            )
        elif optim_type == "lamb":
            sync_node.optimizer = LambSGD(
                sync_node.model_state_dict.values(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                adam=adam,
                amsgrad=amsgrad,
            )
        elif optim_type == "nvlamb":
            from apex.optimizers import FusedLAMB
            sync_node.optimizer = FusedLAMB(
                sync_node.model_state_dict.values(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )
        else:
            raise NotImplementedError("unsupported optimizer type")

        if "lr_schedule" in config:
            sync_node.lrscheduler = CustomStepLR(
                sync_node.optimizer, config["lr_schedule"]
            )
            sync_node.lrscheduler.set_stride(dist.get_world_size())

    def model(self):
        """Return the model"""
        return self.model

    def get_sync_group(self, layer_id=0):
        return self.sync_nodes[layer_id].broadcast_group

    def get_current_lrs(self):
        """Return the current effective lr"""
        lr_arr = []
        lr_arr.append(self.param_groups[0]['lr'])
        for sync_node in self.sync_nodes:
            if sync_node.optimizer:
                lr_arr.append(sync_node.optimizer.param_groups[0]['lr'])
        return lr_arr

    def _refresh_model_state_dicts(self, top_layer_id=None):
        if top_layer_id is None:
            i = len(self.sync_nodes)
        else:
            i = top_layer_id

        # No need to do layer 0
        for x in range(1, i):
            sync_node = self.sync_nodes[x]
            if sync_node.model_state_dict:
                LOG.debug(f"Layer {x} step {self._num_steps} refresh model")
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        sync_node.model_state_dict[name].data.copy_(param.data)

    def _sync_layer_0(self):
        LOG.debug(f"Layer 0 step {self._num_steps} sync")
        self.sync_nodes[0].last_sync_step = self._num_steps

        if (self.sync_nodes[0].optim_type == "ddp" or
            self.sync_nodes[0].optim_type == "apex_ddp"):
            return

        layer0_group = self.sync_nodes[0].broadcast_group
        layer0_comm_size = dist.get_world_size(group=layer0_group)
        handles = []
        for _, param in self.model.named_parameters():
            if param.requires_grad:
                handle = dist.all_reduce(
                    param.grad, group=layer0_group, async_op=True
                )
                handles.append(handle)
        for handle in handles:
            handle.wait()

        for _, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad.div_(layer0_comm_size)

    def _synchronize(self, i, closure=None):
        LOG.debug("_synchronize")
        sync_node = self.sync_nodes[i]

        self._sync_layer_0()

        loss = super(self.__class__, self).step(closure)
        # At this point the model parameters is supposed to be updated,
        # but some code doesn't, like fairseq fp16 optimizer.
        # We need to update the model parameters
        if self._fp16:
            self._sync_fp32_data_to_fp16()



        debugdebugdebug = False
        if debugdebugdebug:
            return loss


        #----------------------------------------
        # Going up layers:
        # 1. each layer get the latest params from self.model, not other layers
        # 2. do all_reduce() and step()
        # 3. copy params back to self.model
        #----------------------------------------
        # Going down:
        # 1. bcast() params from self.model, when applicable
        # 2. each layer refresh their states from self.model, except the top
        #----------------------------------------


        handles = []
        if self.sync_local_optimizer and sync_node.sync_factor > 1:
            handles += self._sync_optimizer()

        # Going up
        for x in range(1, i + 1):
            LOG.debug(f"Layer {x} step {self._num_steps} sync start")

            # step 1 and 2
            node = self.sync_nodes[x]
            if self._is_inverse_pyramid:
                layerx_group = node.broadcast_group
                is_participating = True
            else:
                layerx_group = node.reduce_group
                is_participating = node.my_rank in node.reduce_members
            layerx_comm_size = dist.get_world_size(group=layerx_group)

            if not is_participating:
                break
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if node.model_state_dict:
                        dist_param = node.model_state_dict[name]
                        # Do not do the line below, instead, do inplace to save memory
                        # dist_param.grad = dist_param.data - param.data
                        dist_param.grad.copy_(dist_param.data)
                        dist_param.grad.sub_(param.data)
                        handle = dist.all_reduce(
                            dist_param.grad, group=layerx_group, async_op=True
                        )
                    else:
                        assert(node.optim_type == "avg")
                        # use model's parameter directly
                        handle = dist.all_reduce(
                            param.data, group=layerx_group, async_op=True
                        )
                    handles.append(handle)
            for handle in handles:
                handle.wait()

            layerx_start_time = time.time()
            layerx_sync_time = time.time() - layerx_start_time

            if layerx_sync_time > node.sync_time_threshold:
                LOG.warning(f"Layer {x} step {self._num_steps} sync time above threshold: {layerx_sync_time}")

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if node.model_state_dict:
                        dist_param = node.model_state_dict[name]
                        dist_param.grad.div_(layerx_comm_size)
                    else:
                        assert(node.optim_type == "avg")
                        # use model's parameter directly
                        param.data.div_(layerx_comm_size)

            # step 3
            if node.optimizer:
                if node.lrscheduler is not None:
                    lr = node.lrscheduler.get_lr()
                    LOG.info(f"Layer {x} step {self._num_steps} lr {lr}")
                node.optimizer.step()
                LOG.debug(f"Layer {x} step {self._num_steps} load to model")
                self.model.load_state_dict(node.model_state_dict)
            node.last_sync_step = self._num_steps

            # the model is updated, used to inform callers such as fairseq fp16
            # so they'll skip overriding the model
            self._model_updated = True

        # Going down
        # step 1
        if not self._is_inverse_pyramid and i > 0:
            # barrier?
            LOG.debug(f"Layer {i-1} step {self._num_steps} barrier")
            dist.barrier()

            node = self.sync_nodes[i - 1]
            handles = []
            LOG.debug(f"Layer {i-1} step {self._num_steps} "
                f"broadcast with master node {node.master_rank}")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    handle = dist.broadcast(
                            param.data,
                            node.master_rank,
                            group=node.broadcast_group,
                            async_op=True
                        )
                    handles.append(handle)
            for handle in handles:
                handle.wait()
            
        # step 2
        # model copying to previous layers
        self._refresh_model_state_dicts(i)
        # for x in range(1, i):
        #     node = self.sync_nodes[x]
        #     if node.model_state_dict:
        #         for name in node.model_state_dict:
        #             LOG.debug(f"Layer {x} step {self._num_steps} copy model from top layer")
        #             if sync_node.model_state_dict:
        #                 node.model_state_dict[name].data.copy_(sync_node.model_state_dict[name])
        #             else:
        #                 node.model_state_dict[name].data.copy_(self.model.state_dict[name])

        # At this point the model parameters is up to date.
        # Normally we don't need to update the optimizer, however some code
        # will wrap this step() call and do something extra.
        # e.g fairseq fp16 optimizer, which will overwrite model parameters
        # with stale state in optimizer.
        # We need to update layer 0 / self
        if self._fp16:
            self._sync_fp16_data_to_fp32()

        if self.reset_after_sync and sync_node.sync_factor > 1:
            reset_optimizer(self)

        # if sync_node.sync_factor > 1:
        #     self.model.load_state_dict(sync_node.model_state_dict)

        return loss

    def step(self, closure=None):
        """step"""

        if self._num_steps == 0:
            self._refresh_model_state_dicts()

        self._num_steps += 1

        LOG.debug(f"step {self._num_steps}")
        synced = False
        for i in reversed(range(len(self.sync_nodes))):
            sync_node = self.sync_nodes[i]
            skip = False
            if sync_node.sync_offset > 0 and i < len(self.sync_nodes) - 1:
                steps_since_last_upper_sync = self._num_steps - self.sync_nodes[i+1].last_sync_step
                skip = steps_since_last_upper_sync <= sync_node.sync_offset
            if (not skip and 
                (self._num_steps - sync_node.sync_offset) % sync_node.sync_factor == 0):
                LOG.debug(f"sync at level {i} at step {self._num_steps} begin")

                loss = self._synchronize(i, closure)
                synced = True

                LOG.debug(f"sync at level {i} at step {self._num_steps} end")
                break
        if not synced:  # for num gpu = 1
            loss = super(self.__class__, self).step(closure)

        self._lr_step()
        # self.tb_logger.log_value('Vitals/dist_optim_learning_rate',
        #                                     lr, 
        #                                     step=self._num_steps)

        return loss
    
    def reset_model_updated(self):
        self._model_updated = False

    def model_updated(self):
        return self._model_updated

    def _sync_optimizer(self):
        """_sync_optimizer, local adam momentums"""
        handles = []
        world_size = dist.get_world_size()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                amsgrad = group["amsgrad"] if "amsgrad" in group else False

                state = self.state[p]
                if "exp_avg" not in state:
                    continue
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.div_(world_size)
                exp_avg_sq.div_(world_size)
                handle = dist.all_reduce(exp_avg, async_op=True)
                handles.append(handle)
                handle = dist.all_reduce(exp_avg_sq, async_op=True)
                handles.append(handle)
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    max_exp_avg_sq.div_(world_size)
                    handle = dist.all_reduce(max_exp_avg_sq, async_op=True)
                    handles.append(handle)

        return handles

    def _lr_step(self, i_step=None):
        for sync_node in self.sync_nodes:
            if sync_node.lrscheduler is not None:
                sync_node.lrscheduler.step(i_step)


    # temp hack for fairseq fp16


    def _sync_fp32_data_to_fp16(self):
        LOG.debug("_sync_fp32_data_to_fp16")
        # copy FP32 params back into FP16 model
        offset = 0
        fp32_params = self.param_groups[0]['params'][0]
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            numel = p.data.numel()
            p.data.copy_(fp32_params.data[offset:offset+numel].view_as(p.data))
            offset += numel

    def _sync_fp16_data_to_fp32(self):
        LOG.debug("_sync_fp16_data_to_fp32")
        # copy FP16 params to FP32
        offset = 0
        fp32_params = self.param_groups[0]['params'][0]
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            numel = p.data.numel()
            fp32_params.data[offset:offset+numel].copy_(p.data.view(-1))
            offset += numel

