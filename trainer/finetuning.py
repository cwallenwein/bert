import torch
from torch import nn, optim
from datasets import Dataset
import lightning as L

from model.bert import BertConfig
from trainer.arguments import TrainingArguments
from trainer.scheduler import DynamicWarmupStableDecayScheduler

import wandb
import math
from tqdm import tqdm
from pathlib import Path


class TrainerForSequenceClassificationFinetuning:
    def __init__(self, experiment_name: str, training_args: TrainingArguments, verbose: bool = True):
        self.training_args = training_args
        self.device = self.get_device(training_args.device)
        self.verbose = verbose
        self.experiment_name = experiment_name

    def train(
        self,
        model: L.LightningModule,
        dataset: Dataset,
        epochs: int,
    ):

        # initialize logging
        if self.training_args.with_wandb:
            self.initialize_wandb(model.config, self.training_args)

        # prepare model
        model = model.to(device=self.device, dtype=self.training_args.model_dtype)
        if self.training_args.use_torch_compile and self.device != "mps":
            model = torch.compile(model)
        model.train()

        dataset_size = len(dataset)
        steps = math.ceil(dataset_size / self.training_args.macro_batch_size)

        # prepare dataset
        dataset.set_format("torch", device=self.device)
        steps_per_epoch = dataset_size // self.training_args.macro_batch_size

        # prepare optimizer
        optimizer = model.configure_optimizers()

        # prepare scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=model.learning_rate,
            total_steps=122_719
        )

        # scheduler = DynamicWarmupStableDecayScheduler(
        #     optimizer=optimizer,
        #     lr=model.learning_rate,
        #     warmup_steps=100,
        # )

        for epoch in tqdm(range(epochs)):
            dataset_for_epoch = dataset.iter(batch_size=self.training_args.micro_batch_size)
            for step in tqdm(range(steps)):
                batch = next(dataset_for_epoch)
                batch_idx = epoch * steps_per_epoch + step

                # calculate loss
                loss, metrics = model.training_step(batch, batch_idx)

                # do backward pass
                loss.backward()

                # log loss and lr
                if self.training_args.with_wandb:
                    metrics = metrics | {
                        "loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0]
                    }
                    wandb.log(metrics, step=batch_idx)

                # gradient accumulation
                if self.training_args.use_gradient_clipping:
                    nn.utils.clip_grad_norm_(model.parameters(), self.training_args.gradient_clipping_value)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                progress = (epoch * steps_per_epoch + step) / (epochs * steps_per_epoch)
                if progress >= 0.8 and not scheduler.decaying:
                    scheduler.start_decay(step, 0.8)

        if self.training_args.with_wandb:
            wandb.finish()

        if self.training_args.save_model_after_training:
            self.save_checkpoint(model, optimizer, self.experiment_name, step=dataset_size * epochs)

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
            job_type="finetuning",
            dir="..",
            config={
                "model": model_config.__dict__,
                "training_args": training_args.__dict__,
            },

        )
