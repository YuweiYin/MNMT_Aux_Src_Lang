import sys
from torch.optim.lr_scheduler import _LRScheduler


class CustomStepLR(_LRScheduler):
    """Implement the CNTK style learning rate schedule. The learning rate
    schedule is defined by a string like this:
    0.004*5000:0.002*5000:0.001*4000:0.005. The step here is defined as
    minibatch step. Instead, it is epoch in CNTK.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_scheule_str: learning rate schdule string, e.g.
            0.004*5000:0.002*5000:0.001*4000:0.005
        last_epoch (int): The index of the last batch. This parameter is used
            when resuming a training job. Since `step()` should be invoked
            after each batch instead of after each epoch, this number
            represents the total number of *batches* computed, not the total
            number of epochs computed. When last_epoch=-1, the schedule is
            started from the beginning. Default: -1
    """

    def __init__(self, optimizer, lr_scheule_str, last_epoch=-1):

        self._optimizer = optimizer
        self._lr_scheule_str = lr_scheule_str  # e.g. 0.004*5000:0.002*5000

        # a list of tuples,
        # [(lr, steps_begin, steps_end), (lr, steps_begin, steps_end), ...]
        self._lr_scheule = self._parse_lr_schedule_str(self._lr_scheule_str)

        super(CustomStepLR, self).__init__(optimizer, last_epoch)

    def set_stride(self, stride):
        """Set a stride for each step. This is usful in synchroized multi
        GPU training, for example, horvod. If you use horvod, the stride
        should be the same as the number of GPUs. Because for each step in
        a single worker, it actually include the updates from other workers.

        Args:
            stride: if you set this, whenever you do step(),
                the actual step will be muliplied by stride.
        """

        self._stride = stride  # this is usful when doing multile GPU training.

    def step(self, epoch=None):
        """Move forward 1 step if epoch is None. Move to the step epoch if
        epoch is not None. If stride is not None, move stride steps,
        other wise, move 1 step.

        Arguments:
            epoch: The step you want to move to
        """

        stride = 1
        if hasattr(self, '_stride'):
            stride = self._stride

        if epoch is None:
            self.last_epoch = self.last_epoch + stride
        else:
            self.last_epoch = epoch * stride

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        """Find the learning rate according to the last step from the learning
        rate schedule. It look up the learning rate table one by one. This is
        not optimized solution. But considering the table is usually very
        short, <10 steps, this should be good enough
        """

        for item in self._lr_scheule:
            if self.last_epoch >= item[1] and self.last_epoch < item[2]:
                # use the same lr for each group.
                return [item[0]] * len(self.optimizer.param_groups)

        raise ValueError(
            'Cannot find the lr in this lr {} schedule given step {}'.format(
                self._lr_scheule_str, self.last_epoch))

    def _parse_lr_schedule_str(self, lr_scheule_str):

        # a list of tuples, [(lr, steps), (lr, steps), (lr, steps),...]
        lr_schedule = []
        items = lr_scheule_str.split(':')
        for i in range(len(items)):
            b = items[i].split('*')
            if len(b) == 2:
                lr = float(b[0])
                steps = int(b[1])
            elif len(b) == 1 and i == len(items)-1:
                lr = float(b[0])
                steps = sys.maxsize
            else:
                raise ValueError(
                    'Incorrect format of learning rate scheule {}. A correct \
                    example is: 0.004*5000:0.002*5000:0.001*4000:0.005'
                    .format(lr_scheule_str))

            lr_schedule.append((lr, steps))

        if lr_schedule[-1][1] != sys.maxsize:
            # extend the last schedule to inifinite if not defined.
            lr_schedule.append((lr_schedule[-1][0], sys.maxsize))

        # change to step ranges for convenient lookup
        # [(lr, steps_begin, steps_end), (lr, steps_begin, steps_end), ...]
        range_lr_schedule = []
        range_start = 0
        for item in lr_schedule:
            range_end = range_start+item[1]
            range_lr_schedule.append((item[0], range_start, range_end))
            range_start = range_end

        return range_lr_schedule


class NoamDecayLR(_LRScheduler):
    """Norm decay learning rate schedule. Described in
        https://arxiv.org/pdf/1706.03762.pdf.

        base_value ** (-0.5) * 
            min(step ** (-0.5), step * warmup_steps**(-1.5))

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps: warm up steps
        base_value: base value in the formula
        last_epoch (int): The index of the last batch. This parameter is used
            when resuming a training job. Since `step()` should be invoked
            after each batch instead of after each epoch, this number
            represents the total number of *batches* computed, not the total
            number of epochs computed. When last_epoch=-1, the schedule is
            started from the beginning. Default: -1
    """

    def __init__(self, optimizer, warmup_steps, base_value, last_epoch=-1):

        self._optimizer = optimizer
        self._warmup_steps = warmup_steps
        self._base_value = base_value

        super(NoamDecayLR, self).__init__(optimizer, last_epoch)

    def set_stride(self, stride):
        """Set a stride for each step. This is usful in synchroized multi
        GPU training, for example, horvod. If you use horvod, the stride
        should be the same as the number of GPUs. Because for each step in
        a single worker, it actually include the updates from other workers.

        Args:
            stride: if you set this, whenever you do step(),
                the actual step will be muliplied by stride.
        """

        self._stride = stride  # this is usful when doing multile GPU training.

    def step(self, epoch=None):
        """Move forward 1 step if epoch is None. Move to the step epoch if
        epoch is not None. If stride is not None, move stride steps,
        other wise, move 1 step.

        Arguments:
            epoch: The step you want to move to
        """

        stride = 1
        if hasattr(self, '_stride'):
            stride = self._stride

        if epoch is None:
            self.last_epoch = self.last_epoch + stride
        else:
            self.last_epoch = epoch * stride

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):

        curr_lr = self._base_value ** (-0.5) \
                * min((self.last_epoch+1) ** (-0.5),
                      (self.last_epoch+1) * self._warmup_steps**(-1.5))

        # use the same lr for each group.
        return [curr_lr] * len(self.optimizer.param_groups)
