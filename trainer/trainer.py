import time
import torch
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path
from trainer.arguments import TrainingArguments
from model import BertModelForPretraining
from model.config import BertConfig
from data.save_and_load import get_tokenizer
import torchmetrics
import wandb
from trainer.scheduler import DynamicWarmupStableDecayScheduler
from datasets import Dataset


class TrainerForPreTraining:
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
        model: BertModelForPretraining,
        dataset: Dataset,
        max_steps: int = None,
        max_time_in_min: int = None,
        with_nsp: bool = True,
    ):
        assert max_steps is not None and max_time_in_min is None or max_steps is None and max_time_in_min is not None, "Either max_steps or max_time_in_min must be provided"

        # initialize variables
        total_tokens = 0
        max_time_in_sec = None if max_time_in_min is None else max_time_in_min * 60

        # initialize logging
        if self.training_args.with_wandb:
            self.initialize_wandb(model.config, self.training_args)

        # prepare loss functions
        self.mlm_loss_fn = nn.CrossEntropyLoss()
        if with_nsp:
            self.nsp_loss_fn = nn.BCEWithLogitsLoss()

        # define metrics
        self.mlm_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=model.config.vocab_size,
            average="micro"
        ).to(self.device)
        self.nsp_accuracy = torchmetrics.Accuracy(
            task="binary",
            average="micro"
        ).to(self.device)

        # prepare model
        model = model.to(self.device)
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

        start_time = time.time()
        if max_steps is None:
            progress_bar = tqdm(dataset, desc="Training", unit=" steps")
        else:
            progress_bar = tqdm(dataset, total=max_steps, desc="Training", unit=" steps")

        if max_steps is None:
            max_steps = 10**15
        for step in range(max_steps):
            macro_batch_loss = 0.0
            assert self.training_args.gradient_accumulation_steps > 0, "Gradient accumulation steps must be greater than 0"
            for micro_step in range(self.training_args.gradient_accumulation_steps):
                micro_batch = next(dataset)
                masked_language_modeling_output, next_sentence_prediction_output = model(**micro_batch)

                # calculate loss
                micro_batch_loss = self.calculate_loss(
                    micro_batch, with_nsp, masked_language_modeling_output, next_sentence_prediction_output
                ) / self.training_args.gradient_accumulation_steps
                macro_batch_loss += micro_batch_loss.item()

                # do backward pass
                micro_batch_loss.backward()

                # only log if first micro_batch to reduce overhead
                if micro_step == 0:
                    # calculate mlm and nsp accuracy
                    accuracies = self.calculate_accuracies(
                        micro_batch, with_nsp, masked_language_modeling_output, next_sentence_prediction_output
                    )
                    if self.training_args.with_wandb:
                        wandb.log(accuracies, step=step)

                # count tokens in batch
                total_tokens += self.count_tokens_in_batch(micro_batch, get_tokenizer())

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

            if max_time_in_sec is not None:
                training_time_in_sec = (time.time() - start_time)
                if training_time_in_sec > max_time_in_sec * 0.8 and not scheduler.decaying:
                    scheduler.start_decay(step, 0.8)
                elif training_time_in_sec > max_time_in_sec:
                    break

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

    def calculate_loss(
        self,
        batch,
        with_nsp: bool,
        masked_language_modeling_output,
        next_sentence_prediction_output = None
    ):
        mlm_loss = self.calculate_mlm_loss(batch, masked_language_modeling_output)
        if with_nsp:
            nsp_loss = self.calculate_nsp_loss(batch, next_sentence_prediction_output)
            return mlm_loss + nsp_loss
        else:
            return mlm_loss

    def calculate_accuracies(self, batch, with_nsp: bool, masked_language_modeling_output, next_sentence_prediction_output):
        accuracies = dict()
        accuracies["mlm_acc"] = self.calculate_mlm_acc(batch, masked_language_modeling_output)
        if with_nsp:
            accuracies["nsp_acc"] = self.calculate_nsp_acc(batch, next_sentence_prediction_output)
        return accuracies

    def calculate_mlm_loss(
        self,
        batch,
        masked_language_modeling_output
    ):
        masked_tokens = batch["masked_tokens_mask"].bool()
        masked_token_predictions = masked_language_modeling_output[masked_tokens]
        masked_token_labels = batch["labels"][masked_tokens]

        # calculate loss
        mlm_loss = self.mlm_loss_fn(masked_token_predictions, masked_token_labels)

        return mlm_loss

    def calculate_mlm_acc(self, batch, masked_language_modeling_output):
        masked_tokens = batch["masked_tokens_mask"].bool()
        masked_token_predictions = masked_language_modeling_output[masked_tokens]
        masked_token_labels = batch["labels"][masked_tokens]

        # calculate accuracy
        mlm_acc = self.mlm_accuracy(
            masked_token_predictions, masked_token_labels
        )
        return mlm_acc

    def calculate_nsp_loss(self, batch, next_sentence_prediction_output):
        next_sentence_prediction_labels = batch["labels"][..., 0]

        # calculate loss
        nsp_loss = self.nsp_loss_fn(
            next_sentence_prediction_output, next_sentence_prediction_labels
        )
        return nsp_loss

    def calculate_nsp_acc(self, batch, next_sentence_prediction_output):
        next_sentence_prediction_labels = batch["labels"][..., 0]

        # calculate accuracy
        nsp_acc = self.nsp_accuracy(
            next_sentence_prediction_output, next_sentence_prediction_labels
        )
        return nsp_acc

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
            # set the wandb project where this run will be logged
            project="BERT",

            # track hyperparameters and run metadata
            config={
                "model": model_config.__dict__,
                "training_args": training_args.__dict__,
            }
        )
