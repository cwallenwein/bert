import torch
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path
from trainer.arguments import TrainingArguments
from model import BertModelForPretraining
from model.config import BertConfig
from data.bert_dataset import BertDataset
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
        self.initialize_wandb(model.config, training_args)

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
        with_next_sentence_prediction: bool = True
    ):
        self.masked_language_modeling_accuracy.reset()
        self.next_sentence_prediction_accuracy.reset()

        num_training_samples = self.training_args.batch_size * training_steps
        dataset = BertDataset.load(preprocessed_name=dataset_name, context_length=context_length, verbose=self.verbose)
        dataset.set_format("torch", device=self.device)
        assert num_training_samples <= len(dataset), "Not enough samples in dataset for training steps"
        dataset = dataset.iter(batch_size=self.training_args.batch_size)

        self.model.train()

        progress_bar = tqdm(dataset, total=training_steps, desc="Training", unit=" steps")
        for step in range(training_steps):
            batch = next(dataset)
            total_loss = self.step(batch, with_next_sentence_prediction)
            progress_bar.set_description(f"Training loss: {total_loss: .4f}")
            progress_bar.update(1)

        progress_bar.close()
        wandb.finish()

        if self.training_args.save_model_after_training:
            self.save_checkpoint(experiment_name, step=training_steps)

    def step(self, batch, with_next_sentence_prediction: bool = True):

        self.optimizer.zero_grad()

        masked_language_modeling_output, next_sentence_prediction_output = self.model(**batch)

        masked_language_modeling_loss, masked_language_modeling_accuracy = self.calculate_masked_language_modeling_loss(
            batch, masked_language_modeling_output
        )
        total_loss = masked_language_modeling_loss

        if with_next_sentence_prediction:
            next_sentence_prediction_loss, next_sentence_prediction_accuracy = self.calculate_next_sentence_prediction_loss(
                batch, next_sentence_prediction_output
            )
            total_loss += next_sentence_prediction_loss

        total_loss.backward()
        self.optimizer.step()

        metrics = dict()

        metrics["mlm_loss"] = masked_language_modeling_loss.item()
        metrics["mlm_acc"] = masked_language_modeling_accuracy
        if with_next_sentence_prediction:
            metrics["total_loss"] = total_loss.item()
            metrics["nsp_loss"] = next_sentence_prediction_loss.item()
            metrics["nsp_acc"] = next_sentence_prediction_accuracy

        wandb.log(metrics)
        return total_loss.item()

    def calculate_masked_language_modeling_loss(
        self,
        batch,
        masked_language_modeling_output
    ):
        masked_tokens = batch["masked_tokens"].bool()
        masked_token_predictions = masked_language_modeling_output[masked_tokens]
        masked_token_labels = batch["labels"][masked_tokens]

        # calculate loss
        masked_language_modeling_loss = self.masked_language_modeling_loss_function(masked_token_predictions, masked_token_labels)

        # calculate accuracy
        masked_language_modeling_accuracy = self.masked_language_modeling_accuracy(
            masked_token_predictions, masked_token_labels
        )

        return masked_language_modeling_loss, masked_language_modeling_accuracy

    def calculate_next_sentence_prediction_loss(self, batch, next_sentence_prediction_output):
        next_sentence_prediction_labels = batch["labels"][..., 0]

        if next_sentence_prediction_labels.dtype != torch.float32:
            next_sentence_prediction_labels = next_sentence_prediction_labels.to(torch.float32)

        # calculate loss
        next_sentence_prediction_loss = self.next_sentence_prediction_loss_function(
            next_sentence_prediction_output, next_sentence_prediction_labels
        )

        # calculate accuracy
        next_sentence_prediction_accuracy = self.next_sentence_prediction_accuracy(
            next_sentence_prediction_output, next_sentence_prediction_labels
        )

        return next_sentence_prediction_loss, next_sentence_prediction_accuracy

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
