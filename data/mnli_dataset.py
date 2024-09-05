from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import BertTokenizerFast
import lightning.pytorch as pl
from datasets import load_dataset


class MNLIDataModule(pl.LightningDataModule):
    def __init__(self, context_length: int = 128, batch_size: int = 32, cache_dir="../data/datasets/cache"):
        super().__init__()
        self.context_length = context_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir

    def prepare_data(self) -> None:
        # download
        self.mnli = load_dataset("glue", "mnli", cache_dir=self.cache_dir)
        # tokenize
        self.train_dataset = self.tokenize_mnli(
            self.mnli["train"], self.context_length
        )
        self.validation_dataset = self.tokenize_mnli(
            self.mnli["validation_matched"], self.context_length
        )
        # save to disk

    def setup(self, stage: str):
        # load from disk
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=5)

    @staticmethod
    def tokenize_mnli(dataset: Dataset, context_length: int = 128):
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        def tokenize(sample):
            return tokenizer(
            sample["premise"], sample["hypothesis"],
            max_length=context_length,
            padding="max_length",
            truncation=True
        )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            batch_size=1000,
            remove_columns=[
                "premise",
                "hypothesis",
                "idx"
            ]
        )
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")
        return tokenized_dataset