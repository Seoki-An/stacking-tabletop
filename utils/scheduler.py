import math
from torch.optim.lr_scheduler import _LRScheduler


class BoundedExponentialLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        decay_interval: int,
        decay_rate: float,
        lower_bound: float,
    ):
        self.decay_interval = decay_interval
        self.decay_rate = decay_rate
        self.lower_bound = lower_bound
        super().__init__(optimizer)

    def get_lr(self):
        current_step = self.last_epoch
        lr_multiplier = self.decay_rate ** (current_step / self.decay_interval)
        return [
            max(base_lr * lr_multiplier, self.lower_bound) for base_lr in self.base_lrs
        ]


class WarmupBoundedExponentialLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        decay_interval: int,
        decay_rate: float,
        lower_bound: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.decay_interval = decay_interval
        self.decay_rate = decay_rate
        self.lower_bound = lower_bound
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        if current_step < self.warmup_steps:
            return [
                base_lr * (current_step / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            lr_multiplier = self.decay_rate ** (
                (current_step - self.warmup_steps) / self.decay_interval
            )
            return [
                max(base_lr * lr_multiplier, self.lower_bound)
                for base_lr in self.base_lrs
            ]


class WarmupCosineScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]
