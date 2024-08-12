import time
import torch
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path
from trainer.arguments import TrainingArguments
from model.bert import BertConfig, BertModelForSequenceClassification
import torchmetrics
import wandb
from trainer.scheduler import DynamicWarmupStableDecayScheduler
from datasets import Dataset
import math


class TrainerForSequenceClassificationFinetuning:
    # TODO: reduce the LR after 80% of the training
    def __init__(self, experiment_name: str, training_args: TrainingArguments, verbose: bool = True):
        self.training_args = training_args
        self.device = self.get_device(training_args.device)
        self.verbose = verbose
        self.experiment_name = experiment_name
        self.classification_loss_fn = nn.CrossEntropyLoss()

    def train(
        self,
        model: BertModelForSequenceClassification,
        dataset: Dataset,
        epochs: int,
    ):

        # initialize logging
        if self.training_args.with_wandb:
            self.initialize_wandb(model.config, self.training_args)

        # define metrics
        self.classification_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=model.num_classes,
            average="micro"
        ).to(self.device)

        # prepare model
        model = model.to(self.device)
        if self.training_args.use_torch_compile and self.device != "mps":
            model = torch.compile(model)
        model.train()

        dataset_size = len(dataset)
        steps = math.ceil(dataset_size / self.training_args.macro_batch_size)

        # prepare dataset
        dataset.set_format("torch", device=self.device)

        # prepare optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.training_args.learning_rate,
            betas=(self.training_args.beta1, self.training_args.beta2),
            eps=1e-12
        )

        # prepare scheduler
        scheduler = DynamicWarmupStableDecayScheduler(
            optimizer=optimizer,
            lr=self.training_args.learning_rate,
            warmup_steps=100,
        )

        for epoch in tqdm(range(epochs)):
            dataset = dataset.iter(batch_size=self.training_args.micro_batch_size)
            for step in tqdm(range(steps)):
                macro_batch_loss = 0.0
                assert self.training_args.gradient_accumulation_steps > 0, "Gradient accumulation steps must be greater than 0"
                for micro_step in range(self.training_args.gradient_accumulation_steps):
                    micro_batch = next(dataset)
                    sequence_classification_output = model(**micro_batch)

                    # calculate loss
                    micro_batch_loss = self.classification_loss_fn(
                        sequence_classification_output, micro_batch["labels"]
                    ) / self.training_args.gradient_accumulation_steps
                    macro_batch_loss += micro_batch_loss.item()

                    # do backward pass
                    micro_batch_loss.backward()

                    # only log if first micro_batch to reduce overhead
                    if micro_step == 0:
                        # calculate mlm and nsp accuracy
                        accuracy = self.classification_accuracy(
                            sequence_classification_output, micro_batch["labels"]
                        )
                        if self.training_args.with_wandb:
                            wandb.log({"mnli": accuracy}, step=step)

                # log loss and lr
                if self.training_args.with_wandb:
                    wandb.log({
                        "loss": macro_batch_loss,
                        "learning_rate": scheduler.get_last_lr()[0]
                    }, step=step)

                # gradient accumulation
                if self.training_args.use_gradient_clipping:
                    nn.utils.clip_grad_norm_(model.parameters(), self.training_args.gradient_clipping_value)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                steps_per_epoch = dataset_size // self.training_args.macro_batch_size
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
