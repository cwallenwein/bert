from torch import nn, optim
from tqdm import tqdm
from trainer.arguments import TrainingArguments
from datasets import Dataset

# TODO: make sure to deactivate dropout during eval
class SimpleTrainer:
    def __init__(self, model: nn.Module, training_args: TrainingArguments):
        self.training_args = training_args
        self.model = model

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_args.learning_rate,
            betas=(self.training_args.beta1, self.training_args.beta2)
        )

    def train(self, dataset: Dataset, training_steps: int, long_context_pct: float = 1, long_context_dataset: Dataset = None):
        assert 0 <= long_context_pct <= 1
        if long_context_dataset and long_context_pct > 0:
            training_steps_long = int(training_steps * long_context_pct)
            training_steps_short = training_steps - training_steps_long
            self.train(dataset, training_steps_short)
            self.train(long_context_dataset, training_steps_long)

        num_training_samples = self.training_args.batch_size * training_steps
        assert num_training_samples < len(dataset), "Not enough samples in dataset for training steps"

        self.model.train()
        dataset = dataset.select(range(num_training_samples)).iter(batch_size=self.training_args.batch_size)

        running_loss = None
        progress_bar = tqdm(dataset, total=training_steps, desc="Training", unit=" batches")
        for batch in progress_bar:
            self.optimizer.zero_grad()

            outputs = self.model(**batch)

            masked_token_mask = batch["masked_token_mask"].bool()

            predictions = outputs[masked_token_mask]
            labels = batch["labels"][masked_token_mask]

            loss = self.loss(predictions, labels)
            loss.backward()
            self.optimizer.step()

            if not running_loss:
                running_loss = loss.item()
            else:
                running_loss = 0.5 * loss.item() + 0.5 * running_loss
            progress_bar.set_description(f"Training (loss: {running_loss:.4f})")

        progress_bar.set_description(f"Avg loss: {running_loss:.4f})")
