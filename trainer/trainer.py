import torch
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path
from trainer.arguments import TrainingArguments
from data.bert_dataset import BertDataset


class TrainerForPreTraining:
    def __init__(self, model: nn.Module, training_args: TrainingArguments, verbose: bool = True):
        self.training_args = training_args
        self.model = model

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_args.learning_rate,
            betas=(self.training_args.beta1, self.training_args.beta2)
        )
        self.verbose = verbose

    def train(
        self,
        training_steps: int,
        context_length: int = 128,
        resume_training: bool = False,
        save_training: bool = True
    ):
        num_training_samples = self.training_args.batch_size * training_steps
        dataset = BertDataset.load(num_samples=num_training_samples, context_length=context_length, verbose=self.verbose)
        assert num_training_samples <= len(dataset), "Not enough samples in dataset for training steps"
        dataset = dataset.iter(batch_size=self.training_args.batch_size)

        self.model.train()

        running_total_loss = None
        running_mlm_loss = None
        running_nsp_loss = None
        progress_bar = tqdm(dataset, total=training_steps, desc="Training", unit=" batches")
        for batch in dataset:
            self.optimizer.zero_grad()

            masked_language_modeling_output, next_sentence_prediction_output = self.model(**batch)

            masked_language_modeling_loss = self.calculate_masked_language_modeling_loss(
                batch, masked_language_modeling_output
            )
            next_sentence_prediction_loss = self.calculate_next_sentence_prediction_loss(
                batch, next_sentence_prediction_output
            )
            loss = masked_language_modeling_loss + next_sentence_prediction_loss

            loss.backward()
            self.optimizer.step()

            if not running_total_loss and not running_mlm_loss and not running_nsp_loss:
                running_total_loss = loss.item()
                running_mlm_loss = masked_language_modeling_loss.item()
                running_nsp_loss = next_sentence_prediction_loss.item()
            else:
                running_total_loss = 0.5 * loss.item() + 0.5 * running_total_loss
                running_mlm_loss = 0.5 * masked_language_modeling_loss.item() + 0.5 * running_mlm_loss
                running_nsp_loss = 0.5 * next_sentence_prediction_loss.item() + 0.5 * running_nsp_loss

            progress_bar.set_description(f"Training loss: {running_total_loss: .4f} (MLM: {running_mlm_loss: .4f}, NSP: {running_nsp_loss: .4f})")
            progress_bar.update(1)

        progress_bar.set_description(
            f"Avg loss: {running_total_loss: .4f} (MLM: {running_mlm_loss: .4f}, NSP: {running_nsp_loss: .4f})")
        progress_bar.close()

        if save_training:
            self.save_training_state()

    def calculate_masked_language_modeling_loss(self, batch, masked_language_modeling_output):
        masked_tokens = batch["masked_tokens"].bool()
        masked_token_predictions = masked_language_modeling_output[masked_tokens]
        masked_token_labels = batch["labels"][masked_tokens]
        masked_language_modeling_loss = self.loss_fn(masked_token_predictions, masked_token_labels)
        return masked_language_modeling_loss

    def calculate_next_sentence_prediction_loss(self, batch, next_sentence_prediction_output):
        next_sentence_prediction_labels = batch["labels"][..., 0]
        next_sentence_prediction_loss = self.loss_fn(
            next_sentence_prediction_output, next_sentence_prediction_labels
        )
        return next_sentence_prediction_loss

    def save_training_state(self):
        save_path = Path(__file__).parent.parent / "experiments" / "test.pt"
        print(save_path)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, save_path)
        print(f"Saved training state to {save_path}")

