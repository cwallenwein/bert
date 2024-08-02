import time
import torch
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path
from trainer.arguments import TrainingArguments
from model import BertModelForPretraining
from model.config import BertConfig
from data.bert_dataset import BertDataset
from data.save_and_load import get_tokenizer
import torchmetrics
import wandb
from trainer.scheduler import DynamicWarmupStableDecayScheduler


class TrainerForPreTraining:
    # TODO: log time / step
    # TODO: log total wallclock time
    def __init__(self, model: BertModelForPretraining, training_args: TrainingArguments, verbose: bool = True):
        self.training_args = training_args
        self.device = self.get_device(training_args.device)
        self.verbose = verbose
        self.model = model.to(self.device)

        if training_args.use_torch_compile and self.device != "mps":
            self.model = torch.compile(self.model)

        self.masked_language_modeling_loss_function = nn.CrossEntropyLoss()
        self.next_sentence_prediction_loss_function = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_args.learning_rate,
            betas=(self.training_args.beta1, self.training_args.beta2),
            eps=1e-12
        )

        # Define metrics
        self.mlm_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.model.config.vocab_size,
            average="micro"
        ).to(self.device)
        self.nsp_accuracy = torchmetrics.Accuracy(
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
        with_wandb: bool = True,
        compute_budget_in_h: int = 24
    ):
        # TODO: add micro and macro batch size - make sure that num_gpus * micro * gradient_accumulation_steps == macro_batch_size
        self.total_tokens = 0
        if with_wandb:
            self.initialize_wandb(self.model.config, self.training_args)

        self.mlm_accuracy.reset()
        self.nsp_accuracy.reset()

        num_training_samples = self.training_args.batch_size * training_steps
        dataset = BertDataset.load(preprocessed_name=dataset_name, context_length=context_length, verbose=self.verbose)
        dataset.set_format("torch", device=self.device)
        assert num_training_samples <= len(dataset), "Not enough samples in dataset for training steps"
        dataset = dataset.iter(batch_size=self.training_args.batch_size)

        self.scheduler = DynamicWarmupStableDecayScheduler(
            optimizer=self.optimizer,
            lr=self.training_args.learning_rate,
            warmup_steps=100,
        )

        self.model.train()

        start_time = time.time()
        progress_bar = tqdm(dataset, total=training_steps, desc="Training", unit=" steps")
        for step in range(training_steps):

            macro_batch_loss = 0.0
            assert self.training_args.gradient_accumulation_steps > 0, "Gradient accumulation steps must be greater than 0"
            for micro_step in range(self.training_args.gradient_accumulation_steps):
                micro_batch = next(dataset)
                masked_language_modeling_output, next_sentence_prediction_output = self.model(**micro_batch)

                # calculate loss

                micro_batch_loss = self.calculate_mlm_loss(micro_batch, masked_language_modeling_output) / self.training_args.gradient_accumulation_steps
                if with_nsp:
                    micro_batch_loss += self.calculate_nsp_loss(micro_batch, next_sentence_prediction_output) / self.training_args.gradient_accumulation_steps
                macro_batch_loss += micro_batch_loss.item()

                # do backward pass
                micro_batch_loss.backward()

                # only log if first micro_batch to reduce overhead
                if micro_step == 0:
                    batch_metrics = dict()

                    # calculate accuracy
                    batch_metrics["mlm_acc"] = self.calculate_mlm_acc(micro_batch, masked_language_modeling_output)
                    if with_nsp:
                        batch_metrics["nsp_acc"] = self.calculate_nsp_acc(micro_batch, next_sentence_prediction_output)

                    if with_wandb:
                        wandb.log(batch_metrics, step=step)

                # count tokens in batch
                self.total_tokens += self.count_tokens_in_batch(micro_batch, get_tokenizer())

            # log loss and lr
            if with_wandb:
                wandb.log({
                    "loss": macro_batch_loss,
                    "learning_rate": self.scheduler.get_last_lr()[0]
                }, step=step)

            # gradient accumulation
            if self.training_args.use_gradient_clipping:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.gradient_clipping_value)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if time.time() - start_time > 10 * 60 * 0.8 and not self.scheduler.decaying:
                self.scheduler.start_decay(step, 0.8)
            if time.time() - start_time > 10 * 60:
                break

            progress_bar.set_description(f"Training loss: {macro_batch_loss: .4f}")
            progress_bar.update(1)

        progress_bar.close()
        if with_wandb:
            wandb.finish()

        if self.training_args.save_model_after_training:
            self.save_checkpoint(experiment_name, step=training_steps)

    @staticmethod
    def count_tokens_in_batch(batch, tokenizer):
        # TODO: Count this number after the training
        return (~batch["special_tokens"].bool()).sum()

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
        mlm_acc = self.mlm_accuracy(
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
        nsp_acc = self.nsp_accuracy(
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
