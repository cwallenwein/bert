import torch
from torch import nn, optim
from datasets import Dataset
import lightning as L

from model.bert import BertConfig
from trainer.arguments import TrainingArguments

import wandb
import math
from tqdm import tqdm
from pathlib import Path


class TrainerForSequenceClassificationFinetuning:
    # TODO: define name of run to be name of run for the pretraining+ _finetuning_glue_mnli
    def __init__(self, training_args: TrainingArguments, verbose: bool = True):
        self.training_args = training_args
        self.device = self.get_device(training_args.device)
        self.verbose = verbose

    def train(
        self,
        model: L.LightningModule,
        training_dataset: Dataset,
        validation_dataset: Dataset,
        epochs: int,
        training_steps_per_epoch: int = None,
        validation_steps_per_epoch: int = None
    ):

        # initialize logging
        experiment_name = self.initialize_wandb(model.config, self.training_args)

        # prepare model
        model = model.to(device=self.device, dtype=self.training_args.model_dtype)
        if self.training_args.use_torch_compile and self.device != "mps":
            model = torch.compile(model)

        # prepare dataset
        training_dataset.set_format("torch", device=self.device)
        if training_steps_per_epoch is None:
            training_dataset_size = len(training_dataset)
            training_steps_per_epoch = math.ceil(training_dataset_size / self.training_args.micro_batch_size)
        else:
            training_dataset_size = training_steps_per_epoch * self.training_args.micro_batch_size
            training_dataset = training_dataset.select(range(training_dataset_size))
            training_steps_per_epoch = training_steps_per_epoch
        training_steps_total = training_steps_per_epoch * epochs
        training_global_step = 0

        validation_dataset.set_format("torch", device=self.device)
        if validation_steps_per_epoch is None:
            validation_dataset_size = len(validation_dataset)
            validation_steps_per_epoch = math.ceil(validation_dataset_size / self.training_args.micro_batch_size)
        else:
            validation_dataset_size = validation_steps_per_epoch * self.training_args.micro_batch_size
            validation_dataset = validation_dataset.select(range(validation_dataset_size))
            validation_steps_per_epoch = validation_steps_per_epoch

        validation_steps_total = validation_steps_per_epoch * epochs
        validation_global_step = 0

        # prepare optimizer and scheduler
        optimizer, scheduler = model.configure_optimizers(training_steps_total)

        for epoch in tqdm(range(epochs), desc="Training epochs"):
            model.train()
            iterable_training_dataset = training_dataset.iter(batch_size=self.training_args.micro_batch_size)
            for training_step, training_batch in tqdm(enumerate(iterable_training_dataset), total=training_steps_per_epoch, desc="Training steps"):

                # calculate loss and metrics
                loss, metrics = model.training_step(training_batch, batch_idx=training_step)

                # do backward pass
                loss.backward()

                # log loss and lr
                metrics = metrics | {
                    "train/loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/step": training_global_step,
                    "train/epoch": epoch
                }
                wandb.log(metrics)

                # gradient accumulation
                if self.training_args.use_gradient_clipping:
                    nn.utils.clip_grad_norm_(model.parameters(), self.training_args.gradient_clipping_value)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                progress = training_global_step / training_steps_total
                if progress >= 0.8 and hasattr(scheduler, "decaying") and hasattr(scheduler, "start_decay") and not scheduler.decaying:
                    scheduler.start_decay(current_steps=training_step, training_progress_pct=0.8)

                training_global_step += 1

            # measure validation metrics
            model.eval()
            with torch.no_grad():
                total_validation_accuracy = 0.
                iterable_validation_dataset = validation_dataset.iter(batch_size=self.training_args.micro_batch_size)
                for validation_step, validation_batch in tqdm(enumerate(iterable_validation_dataset), total=validation_steps_per_epoch, desc="Validation steps"):

                    validation_loss, validation_metrics = model.validation_step(validation_batch, batch_idx=validation_step)

                    wandb.log({"val/loss": validation_loss, "val/step": validation_global_step, "val/epoch": epoch})

                    # get average over accuracy
                    total_validation_accuracy += validation_metrics["val/accuracy"]

                    validation_global_step += 1

                avg_validation_accuracy = total_validation_accuracy / validation_steps_per_epoch

                # log validation metrics
                wandb.log({"val/accuracy": avg_validation_accuracy, "val/epoch": epoch})

        wandb.finish()

        if self.training_args.save_model_after_training:
            self.save_checkpoint(model, optimizer, experiment_name, epoch=epochs, step=training_dataset_size)

    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, experiment_name: str, epoch: int, step: int):
        experiment_path = Path(__file__).parent.parent / "experiments" / experiment_name
        experiment_path.mkdir(exist_ok=True, parents=True)
        checkpoint_path = experiment_path / "checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "macro_batch_size": self.training_args.macro_batch_size,
            "config": model.config
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
    def initialize_wandb(model_config: BertConfig, training_args: TrainingArguments) -> str:
        run = wandb.init(
            project="BERT",
            job_type="finetuning",
            dir="..",
            config={
                "model": model_config.__dict__,
                "training_args": training_args.__dict__,
            },
            tags=["mnli"]
        )

        wandb.define_metric("train/step")
        wandb.define_metric("train/epoch", step_metric="train/step")
        wandb.define_metric("train/loss", step_metric="train/step")
        wandb.define_metric("train/accuracy", step_metric="train/step")
        wandb.define_metric("train/learning_rate", step_metric="train/step")

        wandb.define_metric("val/step")
        wandb.define_metric("val/epoch", step_metric="val/step")
        wandb.define_metric("val/loss", step_metric="val/step")
        wandb.define_metric("val/accuracy", step_metric="val/epoch")

        assert run.name is not None
        return run.name
