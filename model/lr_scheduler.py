import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupStableDecayScheduler(LRScheduler):
    """
    Based on https://arxiv.org/pdf/2405.18392
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr: float,
        total_steps: int,
        warmup_steps: int,
        decay_steps: int,
        last_epoch=-1,
        verbose=False,
    ):
        self.lr = lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        lr = self._get_learning_rate(step)
        assert lr > 0, f"Learning rate is {lr}"
        return [lr]

    def _get_learning_rate(self, step: int):
        if step < self.warmup_steps:
            # warmup
            return self.lr * self.linear_warmup_function(step=step)
        elif self.warmup_steps <= step < self.total_steps - self.decay_steps:
            # constant
            return self.lr
        else:
            # decay
            return self.lr * self.sqrt_decay_function(step=step)

    def linear_warmup_function(self, step: int):
        return (step + 1) / self.warmup_steps

    def sqrt_decay_function(self, step: int):
        if step >= self.total_steps:
            step = self.total_steps - 1
        pct_of_decay = 1 - (self.total_steps - step) / self.decay_steps
        return 1 - math.sqrt(pct_of_decay)


class DynamicWarmupStableDecayScheduler(LRScheduler):
    """
    Based on https://arxiv.org/pdf/2405.18392
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr: float,
        warmup_steps: int,
        last_epoch=-1,
        verbose=False,
    ):
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.decaying = False
        self.stage = ""

        super().__init__(optimizer, last_epoch, verbose)

    def start_decay(self, current_steps: int, training_progress_pct: float = 0.8):
        print("Starting decay of learning rate")
        self.decaying = True
        decay_pct = 1 - training_progress_pct
        total_steps = int(current_steps / training_progress_pct)
        assert (
            total_steps > 0
        ), f"current_steps: {current_steps}, training_progress_pct: {training_progress_pct}"
        decay_steps = int(total_steps * decay_pct)
        assert decay_steps > 0, f"total_steps: {total_steps}, decay_pct: {decay_pct}"
        self.decay_steps = decay_steps
        self.total_steps = total_steps

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        lr = self._get_learning_rate(step)
        assert lr > 0, f"Learning rate is {lr}"
        return [lr]

    def _get_learning_rate(self, step: int):
        if step < self.warmup_steps:
            # warmup
            return self.lr * self.linear_warmup_function(step=step)
        elif not self.decaying:
            # constant
            return self.lr
        else:
            # decay
            return self.lr * self.sqrt_decay_function(step=step)

    def linear_warmup_function(self, step: int):
        return (step + 1) / self.warmup_steps

    def sqrt_decay_function(self, step: int):
        if step >= self.total_steps:
            step = self.total_steps - 1
        pct_of_decay = 1 - (self.total_steps - step) / self.decay_steps
        return 1 - math.sqrt(pct_of_decay)
