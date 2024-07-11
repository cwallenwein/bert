import torch
import random
from transformers import PreTrainedTokenizerFast
from datasets import Dataset, concatenate_datasets
from data.not_next_data_processor import NotNextDataProcessor
from data.is_next_data_processor import IsNextDataProcessor


class NextSentencePredictionDataProcessor:

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, dataset: Dataset, num_samples: int, context_length: int = 128, verbose: bool = True):
        num_samples_per_type = num_samples // 2
        k_is_next = num_samples_per_type
        not_next_failure_rate = 0.35
        k_not_next = int(num_samples_per_type // not_next_failure_rate)

        k = k_is_next + k_not_next

        if k < len(dataset):
            # sample without replacement
            random_indices = random.sample(range(len(dataset)), k=k)
        else:
            # sample with replacement
            random_indices = random.choices(range(len(dataset)), k=k)

        split_next = IsNextDataProcessor(
            tokenizer=self.tokenizer, dataset=dataset.select(random_indices[:k_is_next])
        )
        is_next_dataset = split_next(context_length=context_length)
        is_next_dataset = self.add_labels_and_special_tokens_mask(is_next_dataset, label=1)

        split_not_next = NotNextDataProcessor(
            tokenizer=self.tokenizer, dataset=dataset.select(random_indices[k_is_next:])
        )
        not_next_dataset = split_not_next(context_length=context_length, verbose=verbose).select(range(num_samples_per_type))
        not_next_dataset = self.add_labels_and_special_tokens_mask(not_next_dataset, label=0)

        if split_not_next.num_batches > 0:
            print(split_not_next.num_not_possible / split_not_next.num_batches)

        dataset = concatenate_datasets([is_next_dataset, not_next_dataset]).shuffle()
        assert len(dataset) == num_samples
        return dataset

    def add_labels_and_special_tokens_mask(self, dataset: Dataset, label: int):
        return dataset.map(
            lambda batch: self.add_special_tokens_mask(self.add_labels(batch, label)),
            batched=True
        )

    @staticmethod
    def add_labels(batch, label):
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][:, 0] = label
        return batch

    def add_special_tokens_mask(self, batch: dict):
        batch["special_tokens"] = torch.zeros_like(batch["input_ids"])
        batch["special_tokens"][batch["input_ids"] == self.cls_token_id] = 1
        batch["special_tokens"][batch["input_ids"] == self.sep_token_id] = 1
        batch["special_tokens"][batch["input_ids"] == self.pad_token_id] = 1
        return batch
