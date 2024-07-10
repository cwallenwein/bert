from transformers import PreTrainedTokenizerFast
from datasets import Dataset
import random
import torch


class IsNextDataProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, dataset: Dataset):
        assert dataset.column_names == ["input_ids"]
        self.dataset = dataset
        self.column_names = dataset.column_names
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, context_length: int = 128):
        size_before = len(self.dataset)
        dataset = self.dataset.map(
            self.split_is_next,
            batched=True,
            batch_size=2,
            desc=f"Apply is next split",
            num_proc=6,
            input_columns=["input_ids"],
            fn_kwargs={"context_length": context_length},
            # drop_last_batch=True
        ).with_format(
            "torch"
        ).map(
            self.add_labels_and_special_tokens_mask,
            batched=True,
        )
        size_after = len(dataset)
        assert size_after == size_before, f"Dataset size changed from {size_before} to {size_after}"
        return dataset

    def add_labels_and_special_tokens_mask(self, batch):
        batch = self.add_labels(batch, 1)
        batch = self.add_special_tokens_mask(batch)
        return batch

    @staticmethod
    def add_labels(batch, label):
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][:, 0] = label
        return batch

    def add_special_tokens_mask(self, batch: dict):
        batch["special_tokens"] = torch.zeros_like(batch["input_ids"])
        batch["special_tokens"][batch["input_ids"] == self.cls_token_id] = 1
        batch["special_tokens"][batch["input_ids"] == self.sep_token_id] = 1
        return batch

    @staticmethod
    def _generate_split_indices(input_ids):
        batch_num_sentences = [len(sample) for sample in input_ids]
        assert min(batch_num_sentences) >= 2, "Each sample must contain at least two sentences"
        split_indices = [random.randint(1, num_sentences - 1) for num_sentences in batch_num_sentences]
        return split_indices

    def split_is_next(self, input_ids, context_length: int):
        split_indices = self._generate_split_indices(input_ids)
        new_batch = dict()
        input_ids = self._split_input_ids(input_ids, split_indices, context_length=context_length)
        new_batch["input_ids"] = input_ids
        new_batch["token_type_ids"] = self._create_token_type_ids(input_ids)
        new_batch["attention_mask"] = self._create_attention_mask(input_ids)
        return new_batch

    def _split_input_ids(self, input_ids_batch, split_indices, context_length: int):
        new_input_ids_batch = []
        for input_ids, split_idx in zip(input_ids_batch, split_indices):
            new_sample = [self.cls_token_id]
            for sentence_idx, sentence_input_ids in enumerate(input_ids):
                if sentence_idx == split_idx:
                    new_sample.append(self.sep_token_id)
                new_sample.extend(sentence_input_ids)
            new_sample.append(self.sep_token_id)
            assert len(new_sample) <= context_length, f"new_sample has length: {len(new_sample)}"
            new_sample.extend([self.pad_token_id] * (context_length - len(new_sample)))
            new_input_ids_batch.append(new_sample)
        return new_input_ids_batch

    def _create_token_type_ids(self, input_ids_batch: list):
        token_type_ids_batch = []
        for input_ids in input_ids_batch:
            sep_index = input_ids.index(self.sep_token_id)
            token_type_ids = [0] * (sep_index + 1) + [1] * (len(input_ids) - sep_index - 1)
            token_type_ids_batch.append(token_type_ids)
        return token_type_ids_batch

    def _create_attention_mask(self, input_ids_batch):
        attention_mask_batch = []
        for input_ids in input_ids_batch:
            attention_mask = [0 if input_id == self.pad_token_id else 1 for input_id in input_ids]
            attention_mask_batch.append(attention_mask)
        return attention_mask_batch


if __name__ == "__main__":
    from data.util import get_dataset
    from data.context_length_splitter import ContextLengthSplitter
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    raw_dataset = get_dataset(raw=True).select(range(2))

    context_length_splitter = ContextLengthSplitter(tokenizer=tokenizer)
    dataset = context_length_splitter(dataset=raw_dataset, context_length=128, batch_size=2)
    is_next_splitter = IsNextDataProcessor(dataset=dataset, tokenizer=tokenizer)
    next_batch = next(dataset.iter(batch_size=1))
    transformed_batch = is_next_splitter.split_is_next(next_batch, context_length=128)
    print(
        list(zip(
            transformed_batch["input_ids"][0], transformed_batch["attention_mask"][0]
        ))
    )
    print(tokenizer.batch_decode(transformed_batch["input_ids"]))
    print(transformed_batch["input_ids"][0].index(tokenizer.sep_token_id))
    print(transformed_batch.keys())
