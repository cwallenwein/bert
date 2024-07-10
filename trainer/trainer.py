from torch import nn, optim
from tqdm import tqdm
from trainer.arguments import TrainingArguments
from data.bert_dataset import BertDataset


class Trainer:
    def __init__(self, model: nn.Module, training_args: TrainingArguments):
        self.training_args = training_args
        self.model = model

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_args.learning_rate,
            betas=(self.training_args.beta1, self.training_args.beta2)
        )

    def train(self, training_steps: int, context_length: int = 128):
        num_training_samples = self.training_args.batch_size * training_steps
        dataset = BertDataset.load(num_samples=num_training_samples, context_length=context_length)
        assert num_training_samples <= len(dataset), "Not enough samples in dataset for training steps"
        dataset = dataset.iter(batch_size=self.training_args.batch_size)

        self.model.train()

        running_loss = None
        progress_bar = tqdm(dataset, total=training_steps, desc="Training", unit=" batches")
        for batch in dataset:
            self.optimizer.zero_grad()

            outputs = self.model(**batch)

            masked_token_mask = batch["masked_tokens"].bool()

            predictions = outputs[masked_token_mask]
            labels = batch["labels"][masked_token_mask]

            loss = self.loss(predictions, labels)
            loss.backward()
            self.optimizer.step()

            if not running_loss:
                running_loss = loss.item()
            else:
                running_loss = 0.5 * loss.item() + 0.5 * running_loss
            progress_bar.set_description(f"Training (loss: {running_loss: .4f})")
            progress_bar.update(1)

        progress_bar.set_description(f"Avg loss: {running_loss: .4f})")
        progress_bar.close()
