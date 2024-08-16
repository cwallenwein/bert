import torch
from torch import nn, optim
from datasets import Dataset
import lightning as L

from model.bert import BertConfig
from trainer.arguments import TrainingArguments
from data.old.save_and_load import get_tokenizer

import wandb
import time
from tqdm import tqdm
from pathlib import Path

# TODO: merge trainer for pre-training and fine-tuning?


class TrainerForPreTraining:
    # TODO: set bfloat16 for model and optimizer
    # TODO: log time / step
    # TODO: log total wallclock time
    # TODO: log flops / utilization
    # TODO: add lr for max_steps (currently only supported for max_time_in_min)
    def __init__(self, experiment_name: str, training_args: TrainingArguments, verbose: bool = True):
        self.training_args = training_args
        self.device = self.get_device(training_args.device)
        self.verbose = verbose
        self.experiment_name = experiment_name

    def train(
        self,
        model: L.LightningModule,
        dataset: Dataset,
        max_steps: int = None,
        max_epochs: int = None,
        max_time_in_min: int = None,
    ):
        # TODO: prepare everything
        # TODO: depending on max_steps, max_epochs or max_time_in_min decide how to iterate
        # TODO: make inner iteration identical

        if max_steps is not None:
            assert max_epochs is None and max_time_in_min is None
        elif max_epochs is not None:
            assert max_steps is None and max_time_in_min is None
        elif max_time_in_min is not None:
            assert max_steps is None and max_epochs is None

        # initialize variables
        total_tokens = 0
        max_time_in_sec = None if max_time_in_min is None else max_time_in_min * 60

        # initialize logging
        if self.training_args.with_wandb:
            self.initialize_wandb(model.config, self.training_args)

        # prepare model
        model = model.to(device=self.device, dtype=self.training_args.model_dtype)
        if self.training_args.use_torch_compile and self.device != "mps":
            model = torch.compile(model)
        model.train()

        # prepare dataset
        dataset.set_format("torch", device=self.device)
        if max_steps is not None:
            num_training_samples = self.training_args.macro_batch_size * max_steps
            assert num_training_samples <= len(dataset), "Not enough samples in dataset for training steps"
        dataset = dataset.iter(batch_size=self.training_args.micro_batch_size)

        # prepare optimizer
        optimizer = model.configure_optimizers()

        # prepare scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=model.learning_rate,
            total_steps=3_000
        )

        # scheduler = DynamicWarmupStableDecayScheduler(
        #     optimizer=optimizer,
        #     lr=model.learning_rate,
        #     warmup_steps=16,
        # )

        start_time = time.time()
        if max_steps is None:
            progress_bar = tqdm(dataset, desc="Training", unit=" steps")
        else:
            progress_bar = tqdm(dataset, total=max_steps, desc="Training", unit=" steps")

        if max_steps is None:
            max_steps = 10**15
        for step in range(max_steps):
            macro_batch_loss = 0.0
            step_start = time.time()
            assert self.training_args.gradient_accumulation_steps > 0, "Gradient accumulation steps must be greater than 0"
            for micro_step in range(self.training_args.gradient_accumulation_steps):
                micro_batch = next(dataset)

                # calculate loss
                batch_idx = step*self.training_args.gradient_accumulation_steps+micro_step
                micro_batch_loss = model.training_step(micro_batch, batch_idx, step, micro_step, wandb) / self.training_args.gradient_accumulation_steps
                macro_batch_loss += micro_batch_loss.item()

                # do backward pass
                micro_batch_loss.backward()

                # count tokens in batch
                total_tokens += self.count_tokens_in_batch(micro_batch, get_tokenizer())


            # gradient accumulation
            if self.training_args.use_gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), self.training_args.gradient_clipping_value)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if max_time_in_sec is not None:
                training_time_in_sec = (time.time() - start_time)
                if training_time_in_sec > max_time_in_sec * 0.8 and hasattr(scheduler, "decaying") and hasattr(scheduler, "start_decay") and not scheduler.decaying:
                    scheduler.start_decay(step, 0.8)
                elif training_time_in_sec > max_time_in_sec:
                    break

            # log loss, lr and step time
            step_time = time.time() - step_start
            if self.training_args.with_wandb:
                wandb.log({
                    "loss": macro_batch_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "step_duration": step_time,
                    "step_duration_per_sample": step_time / self.training_args.macro_batch_size,

                }, step=step)

            progress_bar.set_description(f"Training loss: {macro_batch_loss: .4f}")
            progress_bar.update(1)

        progress_bar.close()
        if self.training_args.with_wandb:
            wandb.finish()

        if self.training_args.save_model_after_training:
            self.save_checkpoint(model, optimizer, self.experiment_name, step=max_steps)

    @staticmethod
    def count_tokens_in_batch(batch, tokenizer):
        # TODO: Count this number after the training
        return (~batch["special_tokens_mask"].bool()).sum()

    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, experiment_name: str, step: int):
        experiment_path = Path(__file__).parent.parent / "experiments" / experiment_name
        experiment_path.mkdir(exist_ok=True, parents=True)
        checkpoint_path = experiment_path / "checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "macro_batch_size": self.training_args.macro_batch_size
        }, checkpoint_path)
        print(f"Saved training state to {checkpoint_path}")

    @staticmethod
    def get_device(device):
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def initialize_wandb(model_config: BertConfig, training_args: TrainingArguments):
        wandb.init(
            project="BERT",
            job_type="pretraining",
            dir="..",
            config={
                "model": model_config.__dict__,
                "training_args": training_args.__dict__,
            }
        )
