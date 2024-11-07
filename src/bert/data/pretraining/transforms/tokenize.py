import random
from typing import Optional

from transformers import PreTrainedTokenizerBase

from bert.data.util import is_power_of_2
from datasets import Dataset, Features, Sequence, Value


class TokenizeTransform:
    """
    Takes a text dataset and tokenizes it
    Input columns:
        - text

    Output columns:
        - input_ids
        - attention_mask
        - special_tokens_mask

    TODO implement packing
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_samples: int = None,
        context_length: int = 128,
        packing: bool = False,
        batch_size: int = 1_000,
        num_proc: Optional[int] = None,
        keep_in_memory: bool = False,
    ):
        assert not packing, "Packing is not implemented yet"
        assert is_power_of_2(
            context_length
        ), f"context_length({context_length}) must be a power of 2"

        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.context_length = context_length
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.keep_in_memory = keep_in_memory

    def __call__(
        self,
        dataset: Dataset,
    ) -> Dataset:
        assert (
            "text" in dataset.column_names
        ), f"Actual column names: {dataset.column_names}"

        dataset = dataset.map(
            self.tokenize,
            input_columns="text",
            batched=True,
            batch_size=self.batch_size,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
            num_proc=self.num_proc,
            keep_in_memory=self.keep_in_memory,
            fn_kwargs={"context_length": self.context_length},
            features=Features(
                {
                    "attention_mask": Sequence(
                        feature=Value(dtype="bool", id=None),
                        length=self.context_length,
                        id=None,
                    ),
                    "input_ids": Sequence(
                        feature=Value(dtype="int16", id=None),
                        length=self.context_length,
                        id=None,
                    ),
                    "special_tokens_mask": Sequence(
                        feature=Value(dtype="bool", id=None),
                        length=self.context_length,
                        id=None,
                    ),
                }
            ),
        ).with_format("torch")

        if self.num_samples is not None:
            if len(dataset) >= self.num_samples:
                self.downsample_dataset(dataset, self.num_samples)
            else:
                self.upsample_dataset(dataset, self.num_samples)

        assert sorted(dataset.column_names) == [
            "attention_mask",
            "input_ids",
            "special_tokens_mask",
        ], f"Actual column names: {dataset.column_names}"
        return dataset

    def tokenize(self, batch, context_length: int, return_tensors=None):
        result = self.tokenizer(
            batch,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            max_length=context_length,
            padding="max_length",
            return_overflowing_tokens=True,
            truncation=True,
            return_tensors=return_tensors,
        )

        result.pop("overflow_to_sample_mapping")
        return result

    @staticmethod
    def upsample_dataset(dataset, num_samples):
        len_dataset = len(dataset)
        # sample with replacement -> some sample will be drawn multiple times
        random_indices = random.choices(range(len_dataset), k=num_samples)
        dataset = dataset.select(random_indices)
        return dataset

    @staticmethod
    def downsample_dataset(dataset, num_samples):
        len_dataset = len(dataset)
        # sample without replacement -> no sample is drawn twice
        random_indices = random.sample(range(len_dataset), k=num_samples)
        dataset = dataset.select(random_indices)
        return dataset
