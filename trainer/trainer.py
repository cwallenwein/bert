import torch
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path
from trainer.arguments import TrainingArguments
from model import BertModelForPretraining
from model.config import BertConfig
from data.bert_dataset import BertDataset
from data.save_and_load import get_tokenizer
from functools import reduce
import torchmetrics
import wandb


class TrainerForPreTraining:
    def __init__(self, model: BertModelForPretraining, training_args: TrainingArguments, verbose: bool = True):
        self.training_args = training_args
        self.device = self.get_device(training_args.device)
        self.model = model.to(self.device)
        self.verbose = verbose

        self.masked_language_modeling_loss_function = nn.CrossEntropyLoss()
        self.next_sentence_prediction_loss_function = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_args.learning_rate,
            betas=(self.training_args.beta1, self.training_args.beta2)
        )

        # Define metrics
        self.masked_language_modeling_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.model.config.vocab_size,
            average="micro"
        ).to(self.device)
        self.next_sentence_prediction_accuracy = torchmetrics.Accuracy(
            task="binary",
            average="micro"
        ).to(self.device)
        self.total_loss = None

    def train(
        self,
        training_steps: int,
        dataset_name: str,
        experiment_name: str,
        context_length: int = 128,
        with_nsp: bool = True,
        with_wandb: bool = True
    ):
        # TODO: implement gradient accumulation by only applying loss.backward every x steps + dividing the loss by the number of accumulation steps to normalize them
        # TODO: add micro and macro batch size - make sure that num_gpus * micro * gradient_accumulation_steps == macro_batch_size
        self.masked_language_modeling_accuracy.reset()
        self.next_sentence_prediction_accuracy.reset()
        self.total_tokens = 0
        if with_wandb:
            self.initialize_wandb(self.model.config, self.training_args)

        num_training_samples = self.training_args.batch_size * training_steps
        dataset = BertDataset.load(preprocessed_name=dataset_name, context_length=context_length, verbose=self.verbose)
        dataset.set_format("torch", device=self.device)
        assert num_training_samples <= len(dataset), "Not enough samples in dataset for training steps"
        dataset = dataset.iter(batch_size=self.training_args.batch_size)

        self.model.train()

        progress_bar = tqdm(dataset, total=training_steps, desc="Training", unit=" steps")
        for step in range(training_steps):
            batch = next(dataset)
            total_loss = self.step(
                batch,
                step=step,
                total_steps=training_steps,
                with_nsp=with_nsp,
                with_wandb=with_wandb
            )
            progress_bar.set_description(f"Training loss: {total_loss: .4f}")
            progress_bar.update(1)

        progress_bar.close()
        if with_wandb:
            wandb.finish()

        if self.training_args.save_model_after_training:
            self.save_checkpoint(experiment_name, step=training_steps)

    def step(self, batch, step: int, total_steps: int, with_nsp: bool = True, with_wandb: bool = True):

        masked_language_modeling_output, next_sentence_prediction_output = self.model(**batch)
        batch_metrics = dict()

        batch_mlm_loss = self.calculate_mlm_loss(batch, masked_language_modeling_output)
        batch_mlm_acc = self.calculate_mlm_acc(batch, masked_language_modeling_output)

        batch_total_loss = batch_mlm_loss

        batch_metrics["mlm_loss"] = batch_mlm_loss.item()
        batch_metrics["mlm_acc"] = batch_mlm_acc

        if with_nsp:
            batch_nsp_loss = self.calculate_nsp_loss(batch, next_sentence_prediction_output)
            batch_nsp_acc = self.calculate_nsp_acc(batch, next_sentence_prediction_output)
            batch_total_loss += batch_nsp_loss
            batch_metrics["total_loss"] = batch_total_loss.item()
            batch_metrics["nsp_loss"] = batch_nsp_loss.item()
            batch_metrics["nsp_acc"] = batch_nsp_acc

        batch_total_loss.backward()

        # gradient accumulation
        should_accumulate = (step + 1) % self.training_args.gradient_accumulation_steps == 0
        last_step = step + 1 == total_steps
        if should_accumulate or last_step:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.total_tokens += self.count_tokens_in_batch(batch, get_tokenizer())

        if with_wandb:
            wandb.log(batch_metrics)
        return batch_metrics["total_loss"]

    @staticmethod
    def count_tokens_in_batch(batch, tokenizer):
        # TODO: Count this number after the training
        total_number_of_tokens = batch["input_ids"].numel()
        number_of_pad_tokens = (batch["input_ids"] == tokenizer.pad_token_id).sum().item()
        return total_number_of_tokens - number_of_pad_tokens

    def calculate_mlm_loss(
        self,
        batch,
        masked_language_modeling_output
    ):
        masked_tokens = batch["masked_tokens"].bool()
        masked_token_predictions = masked_language_modeling_output[masked_tokens]
        masked_token_labels = batch["labels"][masked_tokens]

        # calculate loss
        mlm_loss = self.masked_language_modeling_loss_function(masked_token_predictions, masked_token_labels)

        return mlm_loss

    def calculate_mlm_acc(self, batch, masked_language_modeling_output):

        masked_tokens = batch["masked_tokens"].bool()
        masked_token_predictions = masked_language_modeling_output[masked_tokens]
        masked_token_labels = batch["labels"][masked_tokens]

        # calculate accuracy
        mlm_acc = self.masked_language_modeling_accuracy(
            masked_token_predictions, masked_token_labels
        )
        return mlm_acc

    def calculate_nsp_loss(self, batch, next_sentence_prediction_output):
        next_sentence_prediction_labels = batch["labels"][..., 0]

        if next_sentence_prediction_labels.dtype != torch.float32:
            next_sentence_prediction_labels = next_sentence_prediction_labels.to(torch.float32)

        # calculate loss
        nsp_loss = self.next_sentence_prediction_loss_function(
            next_sentence_prediction_output, next_sentence_prediction_labels
        )
        return nsp_loss

    def calculate_nsp_acc(self, batch, next_sentence_prediction_output):
        next_sentence_prediction_labels = batch["labels"][..., 0]

        if next_sentence_prediction_labels.dtype != torch.float32:
            next_sentence_prediction_labels = next_sentence_prediction_labels.to(torch.float32)

        # calculate accuracy
        nsp_acc = self.next_sentence_prediction_accuracy(
            next_sentence_prediction_output, next_sentence_prediction_labels
        )
        return nsp_acc

    def save_checkpoint(self, experiment_name: str, step: int):
        experiment_path = Path(__file__).parent.parent / "experiments" / experiment_name
        experiment_path.mkdir(exist_ok=True, parents=True)
        checkpoint_path = experiment_path / "checkpoint.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            "batch_size": self.training_args.batch_size
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
