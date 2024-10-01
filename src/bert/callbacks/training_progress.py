import time
from datetime import timedelta
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT


class AutoSchedulingCallback(pl.callbacks.Callback):
    def __init__(
        self, training_duration: timedelta | dict, exclude_first_batch: bool = True
    ):

        if isinstance(training_duration, dict):
            training_duration = timedelta(**training_duration)

        self.total_training_duration = training_duration.total_seconds()
        self.exceeded_training_duration = False
        self.exclude_first_batch = exclude_first_batch
        self.first_batch_duration = 0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.scheduler = pl_module.lr_schedulers()
        self.training_start = time.time()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.step_start_time = time.time()
        # if batch_idx == 0:
        # start tracking time here because there can be delays
        #   between on_train_start and on_train_batch_start
        #   for example to load the data
        # self.training_start = time.time()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        # track step duration
        step_time = time.time() - self.step_start_time
        batch_size = batch["input_ids"].size(0)

        pl_module.log("train/step_duration", step_time)
        pl_module.log("train/step_duration_per_sample", step_time / batch_size)

        if batch_idx == 0 and self.exclude_first_batch:
            self.first_batch_duration = time.time() - self.training_start
            self.training_start += self.first_batch_duration
            self.total_training_duration -= self.first_batch_duration

        # update training progress
        current_training_duration = time.time() - self.training_start

        self.check_training_duration(current_training_duration)

        training_progress = current_training_duration / self.total_training_duration
        training_progress = min(training_progress, 1.0)

        pl_module.log("train/progress", training_progress)
        # self.scheduler.step(training_progress)

    def check_training_duration(self, current_training_duration: int):
        if current_training_duration > self.total_training_duration:
            # training duration can exceed once because training was not yet stopped
            # if it happens multiple times, somethings wrong
            if self.exceeded_training_duration:
                raise Exception(
                    "training_duration was exceeded but training wasn't stopped"
                )
            else:
                self.exceeded_training_duration = True
